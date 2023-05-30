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

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.filebasedsink import FileBasedSink
import ee
import google.auth
import numpy as np
import requests
from typing import NamedTuple

import landcover.data


# Default values.
NUM_SAMPLES = 10  # TODO: FIND VALUE
MAX_REQUESTS = 20  # default EE request quota
FILE_FORMAT = "numpy"

# Constants.
SCALE = 10000  # meters per pixel
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


class DoFnEE(beam.DoFn):
    """Base DoFn for Earth Engine transforms.

    Initializes Earth Engine once per worker.
    """

    def setup(self) -> None:
        # Get the default credentials to authenticate to Earth Engine.
        credentials, project = google.auth.default(
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/earthengine",
            ]
        )
        # Use the Earth Engine High Volume endpoint.
        #   https://developers.google.com/earth-engine/cloud/highvolume
        ee.Initialize(
            credentials.with_quota_project(None),
            project=project,
            opt_url="https://earthengine-highvolume.googleapis.com",
        )


class SamplePoints(DoFnEE):
    """Selects around the same number of points for every classification.

    This expects the input image to be an integer, for balanced regression points
    you could do `image.int()` to truncate the values into an integer.
    If the values are too large, it might be good to bucketize, for example
    the range is between 0 and ~1000 `image.divide(100).int()` would give ~10 buckets.

    Attributes:
        seed: Random seed to make sure to get different results on different workers.
        num_points: Total number of points to try to get.

    Yields: Tuples of (longitude, latitude) coordinates.
    """

    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def process(self, seed: int) -> Iterator[tuple[float, float]]:
        land_cover = landcover.data.get_land_cover_2020().uint8()
        elevation_bins = (
            landcover.data.get_elevation()
            .clamp(0, MAX_ELEVATION)
            .divide(MAX_ELEVATION)
            .multiply(ELEVATION_BINS - 1)
            .uint8()
        )
        num_points = int(0.5 + self.num_samples / ELEVATION_BINS / LANDCOVER_CLASSES)
        unique_bins = elevation_bins.multiply(ELEVATION_BINS).add(land_cover)
        points = unique_bins.stratifiedSample(
            numPoints=max(1, num_points),
            region=ee.Geometry.MultiPolygon(WORLD_POLYGONS),
            scale=SCALE,
            seed=seed,
            geometries=True,
        )
        for point in points.toList(points.size()).getInfo():
            yield point["geometry"]["coordinates"]


class GetPatch(DoFnEE):
    pass


class GetPatch(beam.DoFn):
    def __init__(self, patch_size: int = 128) -> None:
        self.patch_size = patch_size

    def setup(self) -> None:
        credentials, project = google.auth.default(
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/earthengine",
            ]
        )
        # Use the Earth Engine High Volume endpoint.
        #   https://developers.google.com/earth-engine/cloud/highvolume
        ee.Initialize(
            credentials.with_quota_project(None),
            project=project,
            opt_url="https://earthengine-highvolume.googleapis.com",
        )

    def process(self, request: tuple[list[float], list[dict]]) -> Iterator[np.ndarray]:
        (coords, points) = request
        image = get_image(points)
        yield download_patch(coords, image, SCALE, self.patch_size)


@retry.Retry()
def download_patch(
    coords: tuple[float, float],
    image: ee.Image,
    scale: int,
    patch_size: int,
) -> np.ndarray:
    """Get a training patch centered on the coordinates."""
    point = ee.Geometry.Point(coords)
    # Make a projection to discover the scale in degrees.
    # NOTE: Pass this in so it doesn't get computed every time.
    proj = ee.Projection("EPSG:4326").atScale(scale).getInfo()

    # Get scales out of the transform.
    scale_x = proj["transform"][0]
    scale_y = -proj["transform"][4]
    offset_x = -scale_x * (patch_size + 1) / 2
    offset_y = -scale_y * patch_size / 2

    # Make a request object.
    request = {
        "expression": image,
        "fileFormat": "NPY",
        "grid": {
            "dimensions": {"width": patch_size, "height": patch_size},
            "affineTransform": {
                "scaleX": scale_x,
                "shearX": 0,
                "translateX": coords[0] + offset_x,
                "shearY": 0,
                "scaleY": scale_y,
                "translateY": coords[1] + offset_y,
            },
            "crsCode": proj["crs"],
        },
    }
    return np.load(io.BytesIO(ee.data.computePixels(request)))


def get_example(point: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Gets an (inputs, labels) training example for the year 2020.

    Args:
        point: A (longitude, latitude) pair for the point of interest.

    Returns: An (inputs, labels) pair of NumPy arrays.
    """
    input_image = landcover.data.get_input_image(2020)
    label_image = landcover.data.get_label_image()
    return (
        landcover.data.get_patch(input_image, point, PATCH_SIZE, SCALE),
        landcover.data.get_patch(label_image, point, PATCH_SIZE, SCALE),
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
        inputs = [x for x, _ in self.examples]
        labels = [y for _, y in self.examples]
        np.savez_compressed(file_handle, inputs=inputs, labels=labels)
        return super().close(file_handle)


@beam.ptransform_fn
def CreateDataset(
    pcollection: beam.PCollection[int],
    data_path: str,
    num_samples: int = NUM_SAMPLES,
    max_requests: int = MAX_REQUESTS,
    file_format: str = FILE_FORMAT,
) -> beam.PCollection[str]:
    """Creates an Apache Beam pipeline to create a dataset.

    This fetches data from Earth Engine and creates a TFRecords dataset.
    We use `max_requests` to limit the number of concurrent requests to Earth Engine
    to avoid quota issues. You can request for an increas of quota if you need it.

    Args:
        data_path: Directory path to save the TFRecord files.
        num_points: Total number of points to try to get.
        max_requests: Limit the number of concurrent requests to Earth Engine.
        beam_args: Apache Beam command line arguments to parse as pipeline options.
    """

    examples = (
        pcollection
        | "📌 Sample points" >> beam.FlatMap(sample_points, num_samples / max_requests)
        | "🃏 Reshuffle" >> beam.Reshuffle()
        | "📑 Get examples" >> beam.FlatMap(try_get_example)
    )

    match file_format:
        case "numpy":
            return examples | "📚 Write NumPy" >> beam.io.Write(NumpySink(data_path))

        case "tfrecord":
            return (
                examples
                | "✍️ TFExample serialize" >> beam.MapTuple(serialize_to_tfexample)
                | "📚 Write TFRecords"
                >> beam.io.WriteToTFRecord(
                    f"{data_path}/part", file_name_suffix=".tfrecord.gz"
                )
            )

        case "torch":
            raise NotImplementedError("--file-format=torch")

        case "safetensors":
            raise NotImplementedError("--file-format=safetensors")

        case file_format:
            raise ValueError(f"File format not supported: {file_format}")


def __main__():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        help="Directory path to save the TFRecord files.",
    )
    parser.add_argument(
        "--num-samples",
        default=NUM_SAMPLES,
        type=int,
        help="Total number of points to sample.",
    )
    parser.add_argument(
        "--file-format",
        default=FILE_FORMAT,
        choices=["numpy", "tfrecord", "torch", "safetensor"],
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

    beam_options = PipelineOptions(
        beam_args,
        save_main_session=True,
        pickle_library="cloudpickle",
        direct_num_workers=20,  # direct runner
        direct_running_mode="multi_threading",
    )

    with beam.Pipeline(options=beam_options) as pipeline:
        _ = (
            pipeline
            | "🌱 Make seeds" >> beam.Create(range(args.max_requests))
            | "🗄️ Create dataset"
            >> CreateDataset(
                data_path=args.data_path,
                num_samples=args.num_samples,
                max_requests=args.max_requests,
                file_format=args.file_format,
            )
            | beam.Map(print)
        )


if __name__ == "__main__":
    __main__()
