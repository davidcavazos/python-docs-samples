# Copyright 2022 Google LLC
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

from datetime import datetime, timedelta
import logging
import random
from typing import Iterable, List, NamedTuple, Optional, Tuple
import uuid

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
import numpy as np

import serving.data as data

START_DATE = datetime(2017, 7, 10)
END_DATE = datetime.now() - timedelta(days=1)


class Example(NamedTuple):
    inputs: np.ndarray
    labels: np.ndarray


def sample_labels(
    date: datetime, num_points: int, bounds: data.Bounds
) -> Iterable[Tuple[datetime, data.Point]]:
    data.ee_init()
    image = data.get_label_image(date)
    for lat, lon in data.sample_points(image, num_points, bounds, data.SCALE):
        yield (date, data.Point(lat, lon))


def get_training_example(
    date: datetime, point: data.Point, patch_size: int, num_hours: int = 2
) -> Example:
    data.ee_init()
    return Example(
        data.get_input_sequence(date, point, patch_size, num_hours + 1),
        data.get_label_sequence(date, point, patch_size, num_hours),
    )


def try_get_training_example(
    date: datetime, point: data.Point, patch_size: int = 64, num_hours: int = 2
) -> Iterable[Example]:
    try:
        yield get_training_example(date, point, patch_size, num_hours)
    except Exception as e:
        logging.exception(e)


def write_npz_file(example: Example, file_prefix: str) -> str:
    filename = FileSystems.join(file_prefix, f"{uuid.uuid4()}.npz")
    with FileSystems.create(filename) as f:
        np.savez_compressed(f, inputs=example.inputs, labels=example.labels)
    return filename


def run(
    output_path: str,
    num_dates: int,
    num_points: int,
    bounds: data.Bounds,
    patch_size: int,
    max_requests: int,
    beam_args: Optional[List[str]] = None,
) -> None:
    random_dates = [
        START_DATE + (END_DATE - START_DATE) * random.random() for _ in range(num_dates)
    ]

    beam_options = PipelineOptions(
        beam_args,
        save_main_session=True,
        setup_file="./setup.py",
        # Limit the number of workers to limit the number of requests to Earth Engine.
        max_num_workers=max_requests,
    )
    with beam.Pipeline(options=beam_options) as pipeline:
        (
            pipeline
            | "Random dates" >> beam.Create(random_dates)
            | "Sample labels" >> beam.FlatMap(sample_labels, num_points, bounds)
            | "Reshuffle" >> beam.Reshuffle()
            | "Get example" >> beam.FlatMapTuple(try_get_training_example, patch_size)
            | "Write NPZ files" >> beam.Map(write_npz_file, output_path)
            | "Log files" >> beam.Map(logging.info)
        )


if __name__ == "__main__":
    import argparse

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-dates", type=int, default=20)
    parser.add_argument("--num-points", type=int, default=10)
    parser.add_argument("--west", type=float, default=-125.3)
    parser.add_argument("--south", type=float, default=27.4)
    parser.add_argument("--east", type=float, default=-66.5)
    parser.add_argument("--north", type=float, default=49.1)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--max-requests", type=int, default=20)
    args, beam_args = parser.parse_known_args()

    run(
        output_path=args.output_path,
        num_dates=args.num_dates,
        num_points=args.num_points,
        bounds=data.Bounds(args.west, args.south, args.east, args.north),
        patch_size=args.patch_size,
        max_requests=args.max_requests,
        beam_args=beam_args,
    )
