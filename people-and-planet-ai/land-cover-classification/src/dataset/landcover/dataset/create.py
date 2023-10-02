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
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
import ee
import numpy as np

from landcover.dataset.utils import WriteToNumPy
from landcover.dataset.utils import WriteSchema
from landcover.inputs import ee_init
from landcover.inputs import get_example_image
from landcover.inputs import get_patch
from landcover.inputs import sample_points

# Default values.
NUM_SAMPLES = 10  # TODO: FIND VALUES
MAX_REQUESTS = 20  # default EE maximum concurrent request quota

# Constants.
PATCH_SIZE = 9


class SamplePoints(beam.DoFn):
    """Selects around the same number of points for every classification.

    Init:
        num_samples: Total number of points to sample for each bin.

    Args:
        seed: Number to seed the EE stratified sample random generator.

    Yields: Pairs of (longitude, latitude) coordinates.
    """

    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def setup(self) -> None:
        ee_init()
        return super().setup()

    def process(self, seed: int) -> Iterator[tuple[float, float]]:
        yield from sample_points(seed, self.num_samples, scale=1000)


class GetExample(beam.DoFn):
    """Gets an (inputs, labels) training example for the year 2020.

    Init:
        patch_size: Square patch size in pixels.

    Args:
        point: The (longitude, latitude) coordinates for the point of interest.

    Yields: NumPy structured arrays with shape (width, height) with bands as fields.
    """

    def __init__(self, patch_size: int) -> None:
        self.patch_size = patch_size
        self.image = None
        self.crs = "EPSG:4326"  # https://epsg.io/4326
        self.crs_scale = None

    def setup(self) -> None:
        ee_init()
        self.image = get_example_image()
        proj = ee.Projection(self.crs).atScale(10).getInfo()
        scale_x = proj["transform"][0]
        scale_y = -proj["transform"][4]
        self.crs_scale = (scale_x, scale_y)
        return super().setup()

    def process(self, point: tuple[float, float]) -> Iterator[np.ndarray]:
        yield get_patch(point, self.image, PATCH_SIZE, self.crs, self.crs_scale)


@beam.ptransform_fn
def CreateDataset(
    pcoll: beam.pvalue.PBegin,
    num_samples: int = NUM_SAMPLES,
    max_requests: int = MAX_REQUESTS,
    max_size: int | None = None,
) -> beam.PCollection[str]:
    """Creates an Apache Beam pipeline to create a dataset.

    This fetches data from Earth Engine and creates a TFRecords dataset.
    We use `max_requests` to limit the number of concurrent requests to Earth Engine
    to avoid quota issues. You can request for an increas of quota if you need it.

    Args:
        num_samples: Total number of points to sample for each bin.
        max_requests: Limit the number of concurrent requests to Earth Engine.
        beam_args: Apache Beam command line arguments to parse as pipeline options.
    """

    num_samples_per_seed = max(1, int(num_samples / max_requests))
    samples = (
        pcoll
        | "🌱 Make seeds" >> beam.Create(range(args.max_requests))
        | "📌 Sample points" >> beam.ParDo(SamplePoints(num_samples_per_seed))
    )
    if max_size:
        samples = (
            samples
            | f"Limit {max_size}" >> beam.combiners.Sample.FixedSizeGlobally(max_size)
            | "Flatten" >> beam.FlatMap(lambda xs: xs)
        )

    return (
        samples
        | "📉 Throttle" >> beam.Reshuffle(num_buckets=max_requests)
        | "📨 Get examples" >> beam.ParDo(GetExample(PATCH_SIZE))
        | "📈 Unthrottle" >> beam.Reshuffle()
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        help="Directory path to save the dataset files.",
    )
    parser.add_argument(
        "--tfrecords",
        action=argparse.BooleanOptionalAction,
        help="Set to output files as TFRecords.",
    )
    parser.add_argument(
        "--num-samples",
        default=NUM_SAMPLES,
        type=int,
        help="Number of points to sample for data.",
    )
    parser.add_argument(
        "--max-requests",
        default=MAX_REQUESTS,
        type=int,
        help="Limit the number of concurrent requests to Earth Engine.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        help="Limit the maximum size of the dataset if set.",
    )
    args, beam_args = parser.parse_known_args()

    logging.getLogger().setLevel(logging.INFO)

    beam_options = PipelineOptions(beam_args, pickle_library="cloudpickle")
    with beam.Pipeline(options=beam_options) as pipeline:
        dataset = pipeline | "🗄️ Create dataset" >> CreateDataset(
            num_samples=args.num_samples,
            max_requests=args.max_requests,
            max_size=args.max_size,
        )

        output_prefix = FileSystems.join(args.data_path, "examples")
        if args.tfrecords:
            from landcover.model import serialize

            _ = (
                dataset
                | "✍️ To tf.Example" >> beam.Map(serialize)
                | "📝 Write to TFRecords"
                >> beam.io.WriteToTFRecord(
                    file_path_prefix=output_prefix,
                    file_name_suffix=".tfrecord.gz",
                )
            )
            _ = dataset | "🔑 Write schema" >> WriteSchema(args.data_path)

        else:
            _ = dataset | "📝 Write to NumPy" >> WriteToNumPy(output_prefix)
