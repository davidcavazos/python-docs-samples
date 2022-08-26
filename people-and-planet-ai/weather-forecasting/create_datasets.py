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

from datetime import datetime
from typing import Optional, Tuple
import random

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
import numpy as np

from weather.data import get_training_example, sample_points
from weather.utils import Bounds, Example, Point


def random_date(start_date: datetime, end_date: datetime) -> datetime:
    return start_date + (end_date - start_date) * random.random()


def write_npz_file(point: Point, example: Example, file_prefix: str = "data") -> str:
    date_id = str(point.date.date())
    time_id = str(point.date.time()).replace(":", "-")
    coords_id = f"{point.lat:.2f}@{point.lon:.2f}"
    filename = FileSystems.join(file_prefix, date_id, time_id, coords_id + ".npz")
    with FileSystems.create(filename) as f:
        np.savez_compressed(f, inputs=example.inputs, labels=example.labels)
    return filename


def run(
    output_path: str,
    num_dates: int,
    points_per_date: int,
    bounds: Bounds,
    beam_options: Optional[PipelineOptions] = None,
):
    start_date = datetime(2019, 9, 3, 18, 0)
    end_date = datetime(2019, 9, 3, 18, 30)
    with beam.Pipeline(options=beam_options) as pipeline:
        (
            pipeline
            | "Create date range" >> beam.Create([(start_date, end_date)])
            | "Repeat range" >> beam.FlatMap(lambda x: [x] * num_dates)
            | "Random dates" >> beam.MapTuple(random_date)
            | "Reshuffle dates" >> beam.Reshuffle()
            | "Sample points" >> beam.FlatMap(sample_points, bounds, points_per_date)
            | "Reshuffle points" >> beam.Reshuffle()
            | "Get example" >> beam.Map(get_training_example)
            | "Write NPZ files" >> beam.MapTuple(write_npz_file, output_path)
            | beam.Map(print)
        )


if __name__ == "__main__":
    import argparse
    import logging

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-dates", type=int, default=100)
    parser.add_argument("--points-per-date", type=int, default=100)
    parser.add_argument("--west", type=float, default=-79.1)
    parser.add_argument("--south", type=float, default=25.6)
    parser.add_argument("--east", type=float, default=-76.9)
    parser.add_argument("--north", type=float, default=26.5)
    args, beam_args = parser.parse_known_args()

    beam_options = PipelineOptions(
        beam_args,
        save_main_session=True,
        setup_file="./setup.py",
    )
    run(
        output_path=args.output_path,
        num_dates=args.num_dates,
        points_per_date=args.points_per_date,
        bounds=Bounds(args.west, args.south, args.east, args.north),
        beam_options=beam_options,
    )
