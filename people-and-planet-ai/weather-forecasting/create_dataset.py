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

"""Creates a dataset to train a machine learning model."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import random
from typing import List, Optional
import uuid
from collections.abc import Iterable

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
import ee
import numpy as np
import requests

from serving import data

# Default values.
NUM_DATES = 100
POINTS_PER_CLASS = 100
PATCH_SIZE = 128
MAX_REQUESTS = 20  # default EE request quota

# Point sampling.
START_DATE = datetime(2017, 7, 10)
END_DATE = datetime.now() - timedelta(days=1)
POLYGON = [(-140.0, 60.0), (-140.0, -60.0), (-10.0, -60.0), (-10.0, 60.0)]
MAX_PRECIPITATION = 30  # millimeters


def sample_points(date: datetime, points_per_class: int) -> Iterable[tuple]:
    """Selects around the same number of points for every classification.

    Since our labels are numeric continuous values, we convert them into
    integers within a predifined range. Each integer value is treated
    as a different classification.

    Args:
        date: The date of interest.
        points_per_class: Number of points per classification to pick.

    Yields: (date, lon_lat) pairs.
    """
    data.ee_init()
    image = data.get_labels_image(date).clamp(0, MAX_PRECIPITATION).int()
    points = image.stratifiedSample(
        points_per_class,
        region=ee.Geometry.Polygon(POLYGON),
        scale=data.SCALE,
        geometries=True,
    )
    for point in points.toList(points.size()).getInfo():
        yield (date, point["geometry"]["coordinates"])


def get_training_example(
    date: datetime, point: tuple, patch_size: int = PATCH_SIZE
) -> tuple:
    """Gets an (inputs, labels) training example.

    Args:
        date: The date of interest.
        point: A (longitude, latitude) coordinate.
        patch_size: Size in pixels of the surrounding square patch.

    Returns: An (inputs, labels) pair of NumPy arrays.
    """
    data.ee_init()
    return (
        data.get_inputs_patch(date, point, patch_size),
        data.get_labels_patch(date, point, patch_size),
    )


def try_get_example(
    date: datetime, point: tuple, patch_size: int = PATCH_SIZE
) -> Iterable[tuple]:
    """Wrapper over `get_training_examples` that allows it to simply log errors instead of crashing."""
    try:
        yield get_training_example(date, point, patch_size)
    except requests.exceptions.HTTPError as e:
        logging.exception(e)


def write_npz_file(
    inputs: np.ndarray, labels: np.ndarray, file_prefix: str = "data/"
) -> str:
    """Writes an (inputs, labels) pair into a compressed NumPy file.

    Args:
        inputs: Input data as a NumPy array.
        labels: Label data as a NumPy array.
        file_prefix: Directory path to save files to.

    Returns: The filename of the data file.
    """
    filename = FileSystems.join(file_prefix, f"{uuid.uuid4()}.npz")
    with FileSystems.create(filename) as f:
        np.savez_compressed(f, inputs=inputs, labels=labels)
    logging.info(filename)
    return filename


def run(
    data_path: str,
    num_dates: int = NUM_DATES,
    points_per_class: int = POINTS_PER_CLASS,
    patch_size: int = PATCH_SIZE,
    max_requests: int = MAX_REQUESTS,
    beam_args: Optional[List[str]] = None,
) -> None:
    """Runs an Apache Beam pipeline to create a dataset.

    This fetches data from Earth Engine and writes compressed NumPy files.
    We use `max_requests` to limit the number of concurrent requests to Earth Engine
    to avoid quota issues. You can request for an increas of quota if you need it.

    Args:
        data_path: Directory path to save the TFRecord files.
        num_dates: Number of dates to get training points from.
        points_per_class: Number of points per classification to pick.
        patch_size: Size in pixels of the surrounding square patch.
        max_requests: Limit the number of concurrent requests to Earth Engine.
        beam_args: Apache Beam command line arguments to parse as pipeline options.
    """
    random_dates = [
        START_DATE + (END_DATE - START_DATE) * random.random() for _ in range(num_dates)
    ]

    beam_options = PipelineOptions(
        beam_args,
        save_main_session=True,
        setup_file="./setup.py",
        max_num_workers=max_requests,  # distributed runners
        direct_num_workers=max(max_requests, MAX_REQUESTS),  # direct runner
    )
    with beam.Pipeline(options=beam_options) as pipeline:
        (
            pipeline
            | "📆 Random dates" >> beam.Create(random_dates)
            | "📌 Sample points" >> beam.FlatMap(sample_points, points_per_class)
            | "🃏 Reshuffle" >> beam.Reshuffle()
            | "📑 Get example" >> beam.FlatMapTuple(try_get_example, patch_size)
            | "📚 Write NPZ files" >> beam.MapTuple(write_npz_file, data_path)
        )


if __name__ == "__main__":
    import argparse

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--num-dates", type=int, default=1)
    parser.add_argument("--points-per-class", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--max-requests", type=int, default=20)
    args, beam_args = parser.parse_known_args()

    run(
        data_path=args.data_path,
        num_dates=args.num_dates,
        points_per_class=args.points_per_class,
        patch_size=args.patch_size,
        max_requests=args.max_requests,
        beam_args=beam_args,
    )
