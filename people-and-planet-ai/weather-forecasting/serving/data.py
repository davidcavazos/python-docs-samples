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
import io
from typing import List, NamedTuple

import ee
from google.api_core import exceptions, retry
import google.auth
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import requests

SCALE = 10000
WINDOW = timedelta(hours=1)


class Bounds(NamedTuple):
    west: float
    south: float
    east: float
    north: float


class Point(NamedTuple):
    lat: float
    lon: float


def get_input_image(date: datetime) -> ee.Image:
    start_date = date.isoformat()
    end_date = (date + WINDOW).isoformat()

    # https://developers.google.com/earth-engine/datasets/catalog/NOAA_GOES_16_MCMIPF
    cloud_and_moisture = (
        ee.ImageCollection("NOAA/GOES/16/MCMIPF")
        .filterDate(start_date, end_date)
        .select("CMI_C.*")
        .mosaic()
    )

    # https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V06
    precipitation = (
        ee.ImageCollection("NASA/GPM_L3/IMERG_V06")
        .filterDate(start_date, end_date)
        .select("precipitationCal")
        .mosaic()
    )

    # https://developers.google.com/earth-engine/datasets/catalog/CGIAR_SRTM90_V4
    elevation = ee.Image("CGIAR/SRTM90_V4").select("elevation")

    return ee.Image([cloud_and_moisture, precipitation, elevation])


def get_label_image(date: datetime) -> ee.Image:
    start_date = date.isoformat()
    end_date = (date + WINDOW).isoformat()

    # https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V06
    precipitation = (
        ee.ImageCollection("NASA/GPM_L3/IMERG_V06")
        .filterDate(start_date, end_date)
        .select("precipitationCal")
        .mosaic()
    )

    return precipitation


def ee_init() -> None:
    """Authenticate and initialize Earth Engine with the default credentials."""
    # Use the Earth Engine High Volume endpoint.
    #   https://developers.google.com/earth-engine/cloud/highvolume
    credentials, project = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/earthengine",
        ]
    )
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


def sample_points(
    image: ee.Image, num_points: int, bounds: Bounds, scale: int
) -> np.ndarray:
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
    return np.genfromtxt(io.BytesIO(ee_fetch(url)), delimiter=",", skip_header=1)


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


def get_input_sequence(
    date: datetime, point: Point, patch_size: int, num_hours: int
) -> np.ndarray:
    dates = [date + timedelta(hours=h) for h in range(1 - num_hours, 1)]
    image_sequence = [get_input_image(d) for d in dates]
    return get_patch_sequence(image_sequence, point, patch_size, SCALE)


def get_label_sequence(
    date: datetime, point: Point, patch_size: int, num_hours: int
) -> np.ndarray:
    dates = [date + timedelta(hours=h) for h in range(1, num_hours + 1)]
    image_sequence = [get_label_image(d) for d in dates]
    return get_patch_sequence(image_sequence, point, patch_size, SCALE)
