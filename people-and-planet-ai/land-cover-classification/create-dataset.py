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
from typing import BinaryIO

import landcover.data


# Default values.
NUM_SAMPLES = 10  # TODO: FIND VALUE
MAX_REQUESTS = 20  # default EE request quota
FILE_FORMAT = "numpy"

# Constants.
PATCH_SIZE = 5


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
        num_samples: Total number of points to sample for each bin.
    """

    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def process(self, seed: int) -> Iterator[tuple[float, float]]:
        yield from landcover.data.sample_points(seed, self.num_samples, scale=1000)


class GetExample(DoFnEE):
    """Gets an (inputs, labels) training example for the year 2020.

    Args:
        point: A (longitude, latitude) pair for the point of interest.

    Yields: An (inputs, labels) pair of NumPy arrays.
    """

    def __init__(self, patch_size: int) -> None:
        self.patch_size = patch_size
        self.image = None
        self.crs = "EPSG:4326"  # https://epsg.io/3857
        self.crs_scale = None

    def setup(self) -> None:
        super().setup()  # initialize Earth Engine
        self.image = landcover.data.get_example_image()

        proj = ee.Projection(self.crs).atScale(10).getInfo()
        scale_x = proj["transform"][0]
        scale_y = -proj["transform"][4]
        self.crs_scale = (scale_x, scale_y)

    def process(self, point: tuple[float, float]) -> Iterator[np.ndarray]:
        yield landcover.data.get_patch(
            point, self.image, PATCH_SIZE, self.crs, self.crs_scale
        )


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
            bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(example[name]).numpy()]
            )
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
def WriteToNumPy(
    pcollection: beam.PCollection[np.ndarray], file_path_prefix: str
) -> beam.PCollection[str]:
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
        values = {
            name: torch.from_numpy(batch[name].copy()) for name in batch.dtype.names
        }
        torch.save(values, file_handle)
        return super().close(file_handle)


@beam.ptransform_fn
def WriteToTorch(
    pcollection: beam.PCollection[np.ndarray], file_path_prefix: str
) -> beam.PCollection[str]:
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
        values = {
            name: torch.from_numpy(batch[name].copy()) for name in batch.dtype.names
        }
        byte_data = save(values)
        file_handle.write(byte_data)
        return super().close(file_handle)


@beam.ptransform_fn
def WriteToSafeTensors(
    pcollection: beam.PCollection[np.ndarray], file_path_prefix: str
) -> beam.PCollection[str]:
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
        num_samples: Total number of points to sample for each bin.
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
            return examples | "📚 Write SafeTensors" >> WriteToSafeTensors(
                f"{data_path}/examples"
            )

        case file_format:
            raise ValueError(f"File format not supported: {file_format}")


if __name__ == "__main__":
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
