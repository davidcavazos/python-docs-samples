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
from typing import Dict, Iterable, List, Optional, Tuple
from google.api_core import retry, exceptions
import google.auth
import ee
import pandas as pd
import numpy as np
import io
import requests
from typing import NamedTuple


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


def get_image(
    bands: Dict[str, List[str]], date: datetime, time_window: timedelta
) -> ee.Image:
    end_date = date + time_window
    images = [
        ee.ImageCollection(collection)
        .filterDate(date.isoformat(), end_date.isoformat())
        .select(band_names)
        .mosaic()
        for collection, band_names in bands.items()
    ]
    return ee.Image(images)


def balanced_sample(
    image: ee.Image,
    bounds: Bounds,
    num_points: int,
    scale: int,
    band: Optional[str] = None,
) -> Iterable[Tuple[float, float]]:
    def get_coordinates(point: ee.Feature) -> ee.Feature:
        coords = point.geometry().coordinates()
        return ee.Feature(None, {"lat": coords.get(1), "lon": coords.get(0)})

    region = ee.Geometry.Rectangle(bounds)
    points = (
        image.int()
        .stratifiedSample(num_points, band, region, scale, geometries=True)
        .map(get_coordinates)
    )
    url = points.getDownloadURL("CSV", ["lat", "lon"])
    df = pd.read_csv(io.BytesIO(ee_fetch(url)))
    for point in df.to_dict(orient="records"):
        yield (point["lat"], point["lon"])


def get_sequence(
    p: Point,
    hour_deltas: List[int],
    bands: Dict[str, List[str]],
    patch_size: int,
    scale: int,
    time_window: timedelta,
) -> np.ndarray:
    dates = [p.date + timedelta(hours=h) for h in hour_deltas]
    images = [get_image(bands, date, time_window) for date in dates]
    patches = [get_patch(image, p.lat, p.lon, patch_size, scale) for image in images]
    return np.stack(patches, axis=1)


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


@retry.Retry()
def ee_fetch(url: str) -> bytes:
    # If we get "429: Too Many Requests" errors, it's safe to retry the request.
    # The Retry library only works with `google.api_core` exceptions.
    response = requests.get(url)
    if response.status_code == 429:
        raise exceptions.TooManyRequests(response.text)

    # Still raise any other exceptions to make sure we got valid data.
    response.raise_for_status()
    return response.content
