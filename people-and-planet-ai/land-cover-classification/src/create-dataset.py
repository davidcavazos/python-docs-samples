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
import os

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.filesystems import FileSystems
import ee
import google.auth
import numpy as np
import subprocess

# Default values.
NUM_SAMPLES = 10  # TODO: FIND VALUE
MAX_REQUESTS = 20  # default EE maximum concurrent request quota

# Constants.
PATCH_SIZE = 5
FILE_FORMATS = [
    "numpy",
    "tensorflow",
    "torch",
    "safetensors.numpy",
    "safetensors.tensorflow",
    "safetensors.torch",
]


class DoFnEE(beam.DoFn):
    """Base DoFn for Earth Engine transforms, initializes EE once per worker."""

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

    Args:
        seed: Number to seed the EE stratified sample random generator.

    Yields: Pairs of (longitude, latitude) coordinates.
    """

    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def process(self, seed: int) -> Iterator[tuple[float, float]]:
        import landcover.data

        yield from landcover.data.sample_points(seed, self.num_samples, scale=1000)


class GetExample(DoFnEE):
    """Gets an (inputs, labels) training example for the year 2020.

    Attributes:
        patch_size: Square patch size in pixels.
        image: Earth Engine image to get pixels from.
        crs: Coordinate Reference System code.
        crs_scale: Pair of (scale_x, scale_y) transform for the CRS.

    Args:
        point: The (longitude, latitude) coordinates for the point of interest.

    Yields: NumPy structured arrays with shape (width, height).
    """

    def __init__(self, patch_size: int) -> None:
        self.patch_size = patch_size
        self.image = None
        self.crs = "EPSG:4326"  # https://epsg.io/4326
        self.crs_scale = None

    def setup(self) -> None:
        import landcover.data

        super().setup()  # initialize Earth Engine

        # Initialize these values here since they require
        # Earth Engine to be initialized.
        self.image = landcover.data.get_example_image()
        proj = ee.Projection(self.crs).atScale(10).getInfo()
        scale_x = proj["transform"][0]
        scale_y = -proj["transform"][4]
        self.crs_scale = (scale_x, scale_y)

    def process(self, point: tuple[float, float]) -> Iterator[np.ndarray]:
        import landcover.data

        yield landcover.data.get_patch(
            point, self.image, PATCH_SIZE, self.crs, self.crs_scale
        )


@beam.ptransform_fn
def CreateDataset(
    pcollection: beam.PCollection[int],
    file_format: str,
    data_path: str,
    num_samples: int = NUM_SAMPLES,
    max_requests: int = MAX_REQUESTS,
) -> beam.PCollection[str]:
    """Creates an Apache Beam pipeline to create a dataset.

    This fetches data from Earth Engine and creates a TFRecords dataset.
    We use `max_requests` to limit the number of concurrent requests to Earth Engine
    to avoid quota issues. You can request for an increas of quota if you need it.

    Args:
        file_format: File format to write the dataset files to.
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

    path_prefix = FileSystems.join(data_path, "examples")
    match file_format.split("."):
        case ["numpy"]:
            from io_numpy.writer import WriteToNumPy

            return examples | "📚 Write NumPy" >> WriteToNumPy(path_prefix)

        case ["tensorflow"]:
            from io_tensorflow.writer import WriteToTFRecord

            return examples | "📚 Write TFRecords" >> WriteToTFRecord(path_prefix)

        case ["torch"]:
            from io_torch.writer import WriteToTorch

            return examples | "📚 Write PyTorch" >> WriteToTorch(path_prefix)

        case ["safetensors", tensor_format]:
            from io_safetensors.writer import WriteToSafeTensors

            return examples | "📚 Write SafeTensors" >> WriteToSafeTensors(
                path_prefix, tensor_format
            )

        case file_format:
            raise ValueError(f"File format not supported: {file_format}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_format",
        choices=FILE_FORMATS,
        help="File format to write the dataset files to.",
    )
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
        "--max-requests",
        default=MAX_REQUESTS,
        type=int,
        help="Limit the number of concurrent requests to Earth Engine.",
    )
    args, beam_args = parser.parse_known_args()

    def build_package(name: str, version: str = "1.0.0") -> str:
        filename = os.path.join("build", f"{name}-{version}.tar.gz")
        if os.path.exists(filename):
            return filename
        cmd = [
            "python",
            "-m",
            "build",
            "--sdist",
            os.path.join("src", name),
            "--outdir",
            "build",
        ]
        print(f"Running: {cmd}")
        subprocess.run(cmd, check=True)
        return filename

    match args.file_format.split("."):
        case [name]:
            packages = [
                build_package("landcover-data"),
                build_package(f"io-{name}"),
            ]

        case [name, tensor_format]:
            packages = [
                build_package("landcover-data"),
                build_package(f"io-{name}"),
                build_package(f"io-{tensor_format}"),
            ]

        case _:
            raise ValueError(f"File format not supported: {args.file_format}")

    logging.getLogger().setLevel(logging.INFO)

    # Define the pipeline options.
    beam_options = PipelineOptions(
        beam_args,
        save_main_session=True,
        pickle_library="cloudpickle",
        direct_num_workers=20,  # direct runner
        direct_running_mode="multi_threading",
        requirements_cache="skip",
        extra_packages=packages,
    )

    # Define the pipeline.
    with beam.Pipeline(options=beam_options) as pipeline:
        _ = (
            pipeline
            | "🌱 Make seeds" >> beam.Create(range(args.max_requests))
            | "🗄️ Create dataset"
            >> CreateDataset(
                file_format=args.file_format,
                data_path=args.data_path,
                num_samples=args.num_samples,
                max_requests=args.max_requests,
            )
            | beam.Map(print)
        )
