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
import requests
import logging
import random

import ee
from google.api_core import retry, exceptions
import google.auth
import numpy as np
import pandas as pd

INPUTS = {
    "NOAA/GOES/16/MCMIPF": ["CMI_.*"],
    "NASA/GPM_L3/IMERG_V06": ["precipitationCal"],
}
INPUT_HOUR_DELTAS = [-2, -1, 0]

LABELS = {
    "NASA/GPM_L3/IMERG_V06": ["precipitationCal"],
}
LABEL_HOUR_DELTAS = [1, 6]

SCALE = 10000
IMAGE_TIME_WINDOW = timedelta(days=1)


class Bounds(NamedTuple):
    west: float
    south: float
    east: float
    north: float


class Point(NamedTuple):
    date: datetime
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


def sample_points(date: datetime, bounds: Bounds, num_points: int) -> Iterable[Point]:
    def get_coordinates(point: ee.Feature) -> ee.Feature:
        coords = point.geometry().coordinates()
        return ee.Feature(None, {"lat": coords.get(1), "lon": coords.get(0)})

    ee_init()
    image = get_image(LABELS, date).int()
    points = image.stratifiedSample(
        num_points,
        region=ee.Geometry.Rectangle(bounds),
        scale=SCALE,
        geometries=True,
    )
    url = points.map(get_coordinates).getDownloadURL("CSV", ["lat", "lon"])
    df = pd.read_csv(io.BytesIO(ee_fetch(url)))
    for point in df.to_dict(orient="records"):
        yield Point(date, point["lat"], point["lon"])


def get_training_example(point: Point, patch_size: int) -> Tuple[Point, Example]:
    ee_init()
    inputs = get_input_data(point, patch_size)
    labels = get_label_data(point, patch_size)
    return (point, Example(inputs, labels))


def get_input_data(point: Point, patch_size: int) -> np.ndarray:
    return get_patch_sequence(point, patch_size, INPUT_HOUR_DELTAS, INPUTS, SCALE)


def get_label_data(point: Point, patch_size: int) -> np.ndarray:
    return get_patch_sequence(point, patch_size, LABEL_HOUR_DELTAS, LABELS, SCALE)


def get_image(bands: Dict[str, List[str]], date: datetime) -> ee.Image:
    end_date = date + IMAGE_TIME_WINDOW
    images = [
        ee.ImageCollection(collection)
        .filterDate(date.isoformat(), end_date.isoformat())
        .select(band_names)
        .mosaic()
        for collection, band_names in bands.items()
    ]
    return ee.Image(images)


def get_patch(
    image: ee.Image, lat: float, lon: float, patch_size: int, scale: int
) -> np.ndarray:
    point = ee.Geometry.Point([lon, lat])
    url = image.getDownloadURL(
        {
            "region": point.buffer(scale * patch_size / 2, 1).bounds(1),
            "dimensions": [patch_size, patch_size],
            "format": "NPY",
        }
    )
    data = np.load(io.BytesIO(ee_fetch(url)), allow_pickle=True)
    return np.array([data[field] for field in data.dtype.names], dtype=np.float32)


def get_patch_sequence(
    p: Point,
    patch_size: int,
    hour_deltas: List[int],
    bands: Dict[str, List[str]],
    scale: int,
) -> np.ndarray:
    # TODO: batch sequence into a single ee.Image
    dates = [p.date + timedelta(hours=h) for h in hour_deltas]
    images = [get_image(bands, date) for date in dates]
    patches = [get_patch(image, p.lat, p.lon, patch_size, scale) for image in images]
    return np.stack(patches, axis=1)


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


def random_date(start_date: datetime, end_date: datetime) -> datetime:
    return start_date + (end_date - start_date) * random.random()


def write_npz_file(point: Point, example: Example, file_prefix: str = "data") -> str:
    from apache_beam.io.filesystems import FileSystems

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
    patch_size: int,
    beam_args: Optional[List[str]] = None,
):
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions

    start_date = datetime(2017, 7, 10)
    end_date = datetime.now() - timedelta(days=1)

    beam_options = PipelineOptions(
        beam_args,
        save_main_session=True,
        requirements_file="requirements.txt",
    )
    with beam.Pipeline(options=beam_options) as pipeline:
        (
            pipeline
            | "Create date range" >> beam.Create([(start_date, end_date)])
            | "Repeat range" >> beam.FlatMap(lambda x: [x] * num_dates)
            | "Random dates" >> beam.MapTuple(random_date)
            | "Reshuffle dates" >> beam.Reshuffle()
            | "Sample points" >> beam.FlatMap(sample_points, bounds, points_per_date)
            | "Reshuffle points" >> beam.Reshuffle()
            | "Get example" >> beam.Map(get_training_example, patch_size)
            | "Write NPZ files" >> beam.MapTuple(write_npz_file, output_path)
            | "Log files" >> beam.Map(logging.info)
        )


if __name__ == "__main__":
    import argparse

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-dates", type=int, default=100)
    parser.add_argument("--points-per-date", type=int, default=100)
    parser.add_argument("--west", type=float, default=-125.3)
    parser.add_argument("--south", type=float, default=27.4)
    parser.add_argument("--east", type=float, default=-66.5)
    parser.add_argument("--north", type=float, default=49.1)
    parser.add_argument("--patch-size", type=int, default=64)
    args, beam_args = parser.parse_known_args()

    run(
        output_path=args.output_path,
        num_dates=args.num_dates,
        points_per_date=args.points_per_date,
        bounds=Bounds(args.west, args.south, args.east, args.north),
        patch_size=args.patch_size,
        beam_args=beam_args,
    )
