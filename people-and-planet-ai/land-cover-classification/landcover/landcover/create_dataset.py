# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This creates training and validation datasets for the model.

As inputs it takes a Sentinel-2 image consisting of 13 bands.
Each band contains data for a specific range of the electromagnetic spectrum.
    https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED

As outputs it returns the probabilities of each classification for every pixel.
The land cover labels for the training dataset come from the ESA WorldCover.
    https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100
"""

from __future__ import annotations

from collections.abc import Iterator
import logging
import uuid

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.filesystems import FileSystems
from apache_beam.io.filebasedsink import FileBasedSink

import ee
import numpy as np
import requests


# Default values.
NUM_POINTS = 10  # TODO: FIND VALUE
MAX_REQUESTS = 20  # default EE request quota
FILE_FORMAT = "numpy"

# Constants.
PATCH_SIZE = 5
MAX_ELEVATION = 6000  # found empirically
ELEVATION_BINS = 1  # TODO: CHANGE TO 10
LANDCOVER_CLASSES = 9

# Simple polygons covering most land areas in the world.
WORLD_POLYGONS = [
    # Americas
    [(-33.0, -7.0), (-55.0, 53.0), (-166.0, 65.0), (-68.0, -56.0)],
    # Africa, Asia, Europe
    [
        (74.0, 71.0),
        (166.0, 55.0),
        (115.0, -11.0),
        (74.0, -4.0),
        (20.0, -38.0),
        (-29.0, 25.0),
    ],
    # Australia
    [(170.0, -47.0), (179.0, -37.0), (167.0, -12.0), (128.0, 17.0), (106.0, -29.0)],
]


def sample_points(seed: int, num_points: int) -> Iterator[tuple[float, float]]:
    """Selects around the same number of points for every classification.

    This expects the input image to be an integer, for balanced regression points
    you could do `image.int()` to truncate the values into an integer.
    If the values are too large, it might be good to bucketize, for example
    the range is between 0 and ~1000 `image.divide(100).int()` would give ~10 buckets.

    Args:
        seed: Random seed to make sure to get different results on different workers.
        num_points: Total number of points to try to get.

    Yields: Tuples of (longitude, latitude) coordinates.
    """
    from landcover.data import get_elevation, get_land_cover_2020, SCALE

    land_cover = get_land_cover_2020().int64()
    elevation_bins = (
        get_elevation()
        .clamp(0, MAX_ELEVATION)
        .divide(MAX_ELEVATION)
        .multiply(ELEVATION_BINS - 1)
        .int64()
    )
    unique_bins = elevation_bins.multiply(ELEVATION_BINS).add(land_cover)
    points = unique_bins.stratifiedSample(
        numPoints=max(1, int(0.5 + num_points / ELEVATION_BINS / LANDCOVER_CLASSES)),
        region=ee.Geometry.MultiPolygon(WORLD_POLYGONS),
        scale=SCALE,
        seed=seed,
        geometries=True,
    )
    for point in points.toList(points.size()).getInfo():
        yield point["geometry"]["coordinates"]


def get_example(point: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Gets an (inputs, labels) training example for the year 2020.

    Args:
        point: A (longitude, latitude) pair for the point of interest.

    Returns: An (inputs, labels) pair of NumPy arrays.
    """
    from landcover.data import get_input_image, get_label_image, get_patch, SCALE

    return (
        get_patch(get_input_image(2020), point, PATCH_SIZE, SCALE),
        get_patch(get_label_image(), point, PATCH_SIZE, SCALE),
    )


def try_get_example(
    point: tuple[float, float]
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Wrapper over `get_training_examples` that allows it to simply log errors instead of crashing."""
    try:
        yield get_example(point)
    except (requests.exceptions.HTTPError, ee.ee_exception.EEException) as e:
        logging.error(f"🛑 failed to get example: {point}")
        logging.exception(e)


def serialize_to_tfexample(inputs: np.ndarray, labels: np.ndarray) -> bytes:
    """Serializes inputs and labels NumPy arrays as a tf.Example.

    Both inputs and outputs are expected to be dense tensors, not dictionaries.
    We serialize both the inputs and labels to save their shapes.

    Args:
        inputs: The example inputs as dense tensors.
        labels: The example labels as dense tensors.

    Returns: The serialized tf.Example as bytes.
    """
    import tensorflow as tf

    features = {
        name: tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(data).numpy()])
        )
        for name, data in {"inputs": inputs, "labels": labels}.items()
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


class NumpySink(FileBasedSink):
    def __init__(self, data_path: str) -> None:
        super().__init__(f"{data_path}/part", coder=None, file_name_suffix=".npz")
        self.examples = []

    def write_record(self, file_handle, example: tuple[np.ndarray, np.ndarray]):
        self.examples.append(example)

    def close(self, file_handle):
        inputs = [x for (x, _) in self.examples]
        labels = [y for (_, y) in self.examples]
        np.savez_compressed(file_handle, inputs=inputs, labels=labels)
        return super().close(file_handle)


@beam.ptransform_fn
def WriteToNumpy(examples: beam.PCollection, data_path: str) -> beam.PCollection:
    return examples | beam.io.Write(NumpySink(data_path))


def run(
    data_path: str,
    num_points: int = NUM_POINTS,
    max_requests: int = MAX_REQUESTS,
    file_format: str = FILE_FORMAT,
    beam_args: list[str] | None = None,
) -> None:
    """Runs an Apache Beam pipeline to create a dataset.

    This fetches data from Earth Engine and creates a TFRecords dataset.
    We use `max_requests` to limit the number of concurrent requests to Earth Engine
    to avoid quota issues. You can request for an increas of quota if you need it.

    Args:
        data_path: Directory path to save the TFRecord files.
        num_points: Total number of points to try to get.
        max_requests: Limit the number of concurrent requests to Earth Engine.
        beam_args: Apache Beam command line arguments to parse as pipeline options.
    """

    beam_options = PipelineOptions(
        beam_args,
        save_main_session=True,
        max_num_workers=max_requests,  # distributed runners
        direct_num_workers=max_requests,  # direct runner
        direct_running_mode="multi_threading",
        # disk_size_gb=50,
    )

    with beam.Pipeline(options=beam_options) as pipeline:
        examples = (
            pipeline
            | "🌱 Make seeds" >> beam.Create(range(max_requests))
            | "📌 Sample points"
            >> beam.FlatMap(sample_points, num_points / max_requests)
            | "🃏 Reshuffle" >> beam.Reshuffle()
            | "📑 Get examples" >> beam.FlatMap(try_get_example)
        )

        if file_format == "numpy":
            filenames = examples | "📚 Write NumPy" >> WriteToNumpy(data_path)

        elif file_format == "tfrecord":
            filenames = (
                examples
                | "✍️ TFExample serialize" >> beam.MapTuple(serialize_to_tfexample)
                | "📚 Write TFRecords"
                >> beam.io.WriteToTFRecord(
                    f"{data_path}/part", file_name_suffix=".tfrecord.gz"
                )
            )

        elif file_format == "feather":
            raise NotImplementedError(f"file_format: {file_format}")

        elif file_format == "parquet":
            raise NotImplementedError(f"file_format: {file_format}")

        else:
            raise ValueError(f"File format not supported: {file_format}")

        _ = filenames | beam.Map(print)


def __main__():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        help="Directory path to save the TFRecord files.",
    )
    parser.add_argument(
        "--num-points",
        default=NUM_POINTS,
        type=int,
        help="Total number of points to try to get.",
    )
    parser.add_argument(
        "--file-format",
        default=FILE_FORMAT,
        choices=["numpy", "feather", "parquet", "tfrecord"],
        help="File format to write the dataset files to.",
    )
    parser.add_argument(
        "--max-requests",
        default=MAX_REQUESTS,
        type=int,
        help="Limit the number of concurrent requests to Earth Engine.",
    )
    parser.add_argument(
        "--logging",
        default="WARNING",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Logging level.",
    )
    args, beam_args = parser.parse_known_args()

    logging.getLogger().setLevel(logging.getLevelName(args.logging))

    run(
        data_path=args.data_path,
        num_points=args.num_points,
        max_requests=args.max_requests,
        file_format=args.file_format,
        beam_args=beam_args,
    )


if __name__ == "__main__":
    __main__()
