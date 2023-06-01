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

from collections.abc import Callable, Iterator
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.filebasedsink import FileBasedSink
import ee
import google.auth
import numpy as np
import requests
from typing import Any, BinaryIO

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

    Attributes:
        num_points: Total number of points to try to get.

    """

    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def process(self, seed: int) -> Iterator[tuple[float, float]]:
        """

        This expects the input image to be an integer, for balanced regression points
        you could do `image.int()` to truncate the values into an integer.
        If the values are too large, it might be good to bucketize, for example
        the range is between 0 and ~1000 `image.divide(100).int()` would give ~10 buckets.

        Args:
            seed: Random seed to make sure to get different results on different workers.

        Yields: Tuples of (longitude, latitude) coordinates.
        """
        land_cover = landcover.data.get_land_cover().select('landcover')
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


class GetExample(DoFnEE):
    """Gets an (inputs, labels) training example for the year 2020.

    Args:
        point: A (longitude, latitude) pair for the point of interest.

    Yields: An (inputs, labels) pair of NumPy arrays.
    """

    def __init__(self, patch_size: int) -> None:
        self.patch_size = patch_size
        self.crs = "EPSG:4326"
        self.proj = None

    def setup(self) -> None:
        super().setup()
        self.proj = ee.Projection(self.crs).atScale(SCALE).getInfo()

    def process( self, point: tuple[float, float]) -> Iterator[np.ndarray]:
        image = landcover.data.get_example_image()
        crs_scale = (self.proj["transform"][0], self.proj["transform"][4])
        yield landcover.data.get_patch(point, image, PATCH_SIZE, self.crs, crs_scale)


def serialize_to_tfexample(example: np.ndarray) -> bytes:
    """Serializes an example NumPy array as a tf.Example.

    Both inputs and outputs are expected to be dense tensors, not dictionaries.
    We serialize both the inputs and labels to save their shapes.

    Args:
        example: The example array containing inputs and labels.

    Returns: The serialized tf.Example as bytes.
    """
    import tensorflow as tf

    features = {
        name: tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(example[name]).numpy()])
        )
        for name in example.dtype.names
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


class NumPySink(FileBasedSink):
    def __init__(self, file_path_prefix: str) -> None:
        super().__init__(file_path_prefix, coder=None, file_name_suffix=".npz")
        self.examples = []

    def write_record(self, file_handle: BinaryIO, example: np.ndarray):
        self.examples.append(example)

    def close(self, file_handle: BinaryIO):
        batch = np.stack(self.examples)
        values = {name: batch[name] for name in batch.dtype.names}
        np.savez_compressed(file_handle, **values)
        return super().close(file_handle)

@beam.ptransform_fn
def WriteToNumPy(pcollection: beam.PCollection[np.ndarray], file_path_prefix: str) -> beam.PCollection[str]:
    return pcollection | beam.io.Write(NumPySink(file_path_prefix))

class TorchSink(FileBasedSink):
    def __init__(self, file_path_prefix: str) -> None:
        super().__init__(file_path_prefix, coder=None, file_name_suffix=".pt")
        self.examples = []

    def write_record(self, file_handle: BinaryIO, example: np.ndarray):
        self.examples.append(example)

    def close(self, file_handle: BinaryIO):
        import torch

        batch = np.stack(self.examples)
        values = {name: torch.from_numpy(batch[name].copy()) for name in batch.dtype.names}
        torch.save(values, file_handle)
        return super().close(file_handle)

@beam.ptransform_fn
def WriteToTorch(pcollection: beam.PCollection[np.ndarray], file_path_prefix: str) -> beam.PCollection[str]:
    return pcollection | beam.io.Write(TorchSink(file_path_prefix))

class SafeTensorsSink(FileBasedSink):
    def __init__(self, file_path_prefix: str) -> None:
        super().__init__(file_path_prefix, coder=None, file_name_suffix=".safetensors")
        self.examples = []

    def write_record(self, file_handle: BinaryIO, example: np.ndarray):
        self.examples.append(example)

    def close(self, file_handle: BinaryIO):
        from safetensors.torch import save
        import torch

        batch = np.stack(self.examples)
        values = {name: torch.from_numpy(batch[name].copy()) for name in batch.dtype.names}
        byte_data = save(values)
        file_handle.write(byte_data)
        return super().close(file_handle)

@beam.ptransform_fn
def WriteToSafeTensors(pcollection: beam.PCollection[np.ndarray], file_path_prefix: str) -> beam.PCollection[str]:
    return pcollection | beam.io.Write(SafeTensorsSink(file_path_prefix))


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

    num_samples_per_seed = max(1, int(num_samples / max_requests))
    examples = (
        pcollection
        | "📌 Sample points" >> beam.ParDo(SamplePoints(num_samples_per_seed))
        | "📉 Throttle" >> beam.Reshuffle(num_buckets=max_requests)
        | "📨 Get examples" >> beam.ParDo(GetExample(PATCH_SIZE))
        | "📈 Unthrottle" >> beam.Reshuffle()
    )

    match file_format:
        case "numpy":
            return examples | "📚 Write NumPy" >> WriteToNumPy(f"{data_path}/examples")

        case "tfrecord":
            return (
                examples
                | "✍️ TFExample serialize" >> beam.Map(serialize_to_tfexample)
                | "📚 Write TFRecords"
                >> beam.io.WriteToTFRecord(
                    file_path_prefix=f"{data_path}/examples",
                    file_name_suffix=".tfrecord.gz",
                )
            )

        case "torch":
            return examples | "📚 Write PyTorch" >> WriteToTorch(f"{data_path}/examples")

        case "safetensors":
            return examples | "📚 Write SafeTensors" >> WriteToSafeTensors(f"{data_path}/examples")

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
        choices=["numpy", "tfrecord", "torch", "safetensors"],
        help="File format to write the dataset files to.",
    )
    parser.add_argument(
        "--max-requests",
        default=MAX_REQUESTS,
        type=int,
        help="Limit the number of concurrent requests to Earth Engine.",
    )
    args, beam_args = parser.parse_known_args()

    logging.getLogger().setLevel(logging.INFO)

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
