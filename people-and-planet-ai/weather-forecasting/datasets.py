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
from typing import Dict, Iterable, List, Optional, NamedTuple, Tuple
import io
import logging
import random
import requests
import uuid

import ee
from google.api_core import retry, exceptions
import google.auth
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import pandas as pd

INPUTS = {
    "NOAA/GOES/16/MCMIPF": [f"CMI_C{i:02}" for i in range(1, 16 + 1)],
    "NASA/GPM_L3/IMERG_V06": ["precipitationCal"],
}

LABELS = {
    "NASA/GPM_L3/IMERG_V06": ["precipitationCal"],
}

SCALE = 10000
WINDOW = timedelta(hours=1)

START_DATE = datetime(2017, 7, 10)
END_DATE = datetime.now() - timedelta(days=1)


class Bounds(NamedTuple):
    west: float
    south: float
    east: float
    north: float


class Point(NamedTuple):
    lat: float
    lon: float


class Example(NamedTuple):
    inputs: np.ndarray
    labels: np.ndarray


def ee_init() -> None:
    """Authenticate and initialize Earth Engine with the default credentials."""
    # Use the Earth Engine High Volume endpoint.
    #   https://developers.google.com/earth-engine/cloud/highvolume
    credentials, project = google.auth.default()
    ee.Initialize(
        credentials.with_quota_project(None),
        project=project,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )


@retry.Retry(deadline=60 * 20)  # seconds
def ee_fetch(url: str) -> bytes:
    # If we get "429: Too Many Requests" errors, it's safe to retry the request.
    # The Retry library only works with `google.api_core` exceptions.
    response = requests.get(url)
    if response.status_code == 429:
        raise exceptions.TooManyRequests(response.text)

    # Still raise any other exceptions to make sure we got valid data.
    response.raise_for_status()
    return response.content


def get_image(
    date: datetime, bands_schema: Dict[str, List[str]], window: timedelta
) -> ee.Image:
    ee_init()
    images = [
        ee.ImageCollection(collection)
        .filterDate(date.isoformat(), (date + window).isoformat())
        .select(bands)
        .mosaic()
        for collection, bands in bands_schema.items()
    ]
    return ee.Image(images)


def sample_labels(
    date: datetime, num_points: int, bounds: Bounds
) -> Iterable[Tuple[datetime, Point]]:
    image = get_image(date, LABELS, WINDOW)
    for point in sample_points(image, num_points, bounds, SCALE):
        yield (date, point)


def sample_points(
    image: ee.Image, num_points: int, bounds: Bounds, scale: int
) -> Iterable[Point]:
    def get_coordinates(point: ee.Feature) -> ee.Feature:
        coords = point.geometry().coordinates()
        return ee.Feature(None, {"lat": coords.get(1), "lon": coords.get(0)})

    points = image.int().stratifiedSample(
        num_points,
        region=ee.Geometry.Rectangle(bounds),
        scale=scale,
        geometries=True,
    )
    url = points.map(get_coordinates).getDownloadURL("CSV", ["lat", "lon"])
    df = pd.read_csv(io.BytesIO(ee_fetch(url)))
    for point in df.to_dict(orient="records"):
        yield Point(point["lat"], point["lon"])


def get_input_sequence(
    date: datetime, point: Point, patch_size: int, num_hours: int
) -> np.ndarray:
    dates = [date + timedelta(hours=h) for h in range(1 - num_hours, 1)]
    images = [get_image(d, INPUTS, WINDOW) for d in dates]
    return get_patch_sequence(images, point, patch_size, SCALE)


def get_label_sequence(
    date: datetime, point: Point, patch_size: int, num_hours: int
) -> np.ndarray:
    dates = [date + timedelta(hours=h) for h in range(1, num_hours + 1)]
    images = [get_image(d, LABELS, WINDOW) for d in dates]
    return get_patch_sequence(images, point, patch_size, SCALE)


def get_training_example(date: datetime, point: Point, patch_size: int = 64) -> Example:
    ee_init()
    return Example(
        get_input_sequence(date, point, patch_size, 3),
        get_label_sequence(date, point, patch_size, 2),
    )


def get_patch_sequence(
    image_sequence: List[ee.Image], point: Point, patch_size: int, scale: int
) -> np.ndarray:
    def unpack(arr: np.ndarray, i: int) -> np.ndarray:
        names = [x for x in arr.dtype.names if x.startswith(f"{i}_")]
        return np.moveaxis(structured_to_unstructured(arr[names]), -1, 0)

    point = ee.Geometry.Point([point.lon, point.lat])
    image = ee.ImageCollection(image_sequence).toBands()
    url = image.getDownloadURL(
        {
            "region": point.buffer(scale * patch_size / 2, 1).bounds(1),
            "dimensions": [patch_size, patch_size],
            "format": "NPY",
        }
    )
    flat_seq = np.load(io.BytesIO(ee_fetch(url)), allow_pickle=True)
    return np.stack([unpack(flat_seq, i) for i, _ in enumerate(image_sequence)], axis=1)


def write_npz_file(example: Example, file_prefix: str) -> str:
    from apache_beam.io.filesystems import FileSystems

    filename = FileSystems.join(file_prefix, f"{uuid.uuid4()}.npz")
    with FileSystems.create(filename) as f:
        np.savez_compressed(f, inputs=example.inputs, labels=example.labels)
    return filename


def run(
    output_path: str,
    num_dates: int,
    num_points: int,
    bounds: Bounds,
    patch_size: int,
    max_requests: int,
    beam_args: Optional[List[str]] = None,
):
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions

    random_dates = [
        START_DATE + (END_DATE - START_DATE) * random.random() for _ in range(num_dates)
    ]

    beam_options = PipelineOptions(
        beam_args,
        save_main_session=True,
        requirements_file="requirements.txt",
        max_num_workers=max_requests,
    )
    with beam.Pipeline(options=beam_options) as pipeline:
        (
            pipeline
            | "Random dates" >> beam.Create(random_dates)
            | "Sample labels" >> beam.FlatMap(sample_labels, num_points, bounds)
            | "Reshuffle" >> beam.Reshuffle()
            | "Get example" >> beam.MapTuple(get_training_example, patch_size)
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
        bounds=Bounds(args.west, args.south, args.east, args.north),
        patch_size=args.patch_size,
        max_requests=args.max_requests,
        beam_args=beam_args,
    )
