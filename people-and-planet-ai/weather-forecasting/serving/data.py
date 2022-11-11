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

"""Data utilities to grab data from Earth Engine.
Meant to be used for both training and prediction so the model is
trained on exactly the same data that will be used for predictions.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import io

import ee
from google.api_core import exceptions, retry
import google.auth
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import requests

# Constants.
SCALE = 10000  # meters per pixel
INPUT_HOUR_DELTAS = [-4, -2, 0]
OUTPUT_HOUR_DELTAS = [2, 6]
WINDOW = timedelta(hours=1)


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


# https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V06
def get_gpm(date: datetime) -> ee.Image:
    window_start = date.isoformat()
    window_end = (date + WINDOW).isoformat()
    return (
        ee.ImageCollection("NASA/GPM_L3/IMERG_V06")
        .filterDate(window_start, window_end)
        .select("precipitationCal")
        .mosaic()
        .unmask(0)
        .float()
    )


def get_gpm_sequence(dates: list[datetime]) -> ee.Image:
    images = [get_gpm(date) for date in dates]
    return ee.ImageCollection(images).toBands()


# https://developers.google.com/earth-engine/datasets/catalog/NOAA_GOES_16_MCMIPF
def get_goes16(date: datetime) -> ee.Image:
    window_start = date.isoformat()
    window_end = (date + WINDOW).isoformat()
    return (
        ee.ImageCollection("NOAA/GOES/16/MCMIPF")
        .filterDate(window_start, window_end)
        .select("CMI_C.*")
        .mosaic()
        .unmask(0)
        .float()
    )


def get_goes16_sequence(dates: list[datetime]) -> ee.Image:
    images = [get_goes16(date) for date in dates]
    return ee.ImageCollection(images).toBands()


# https://developers.google.com/earth-engine/datasets/catalog/MERIT_DEM_v1_0_3
def get_elevation() -> ee.Image:
    return ee.Image("MERIT/DEM/v1_0_3").rename("elevation").unmask(0).float()


def get_inputs_image(date: datetime) -> ee.Image:
    dates = [date + timedelta(hours=h) for h in INPUT_HOUR_DELTAS]
    precipitation = get_gpm_sequence(dates)
    cloud_and_moisture = get_goes16_sequence(dates)
    elevation = get_elevation()
    return ee.Image([precipitation, cloud_and_moisture, elevation])


def get_labels_image(date: datetime) -> ee.Image:
    dates = [date + timedelta(hours=h) for h in OUTPUT_HOUR_DELTAS]
    return get_gpm_sequence(dates)


def get_inputs_patch(date: datetime, point: tuple, patch_size: int) -> np.ndarray:
    """Gets the patch of pixels for the inputs.

    args:
        date: The date of interest.
        point: A (longitude, latitude) coordinate.
        patch_size: Size in pixels of the surrounding square patch.

    Returns: The pixel values of a patch as a NumPy array.
    """
    image = get_inputs_image(date)
    patch = get_patch(image, point, patch_size, SCALE)
    return structured_to_unstructured(patch)


def get_labels_patch(date: datetime, point: tuple, patch_size: int) -> np.ndarray:
    """Gets the patch of pixels for the labels.

    args:
        date: The date of interest.
        point: A (longitude, latitude) coordinate.
        patch_size: Size in pixels of the surrounding square patch.

    Returns: The pixel values of a patch as a NumPy array.
    """
    image = get_labels_image(date)
    patch = get_patch(image, point, patch_size, SCALE)
    return structured_to_unstructured(patch)


@retry.Retry(deadline=10 * 60)  # seconds
def get_patch(
    image: ee.Image, lonlat: tuple[float, float], patch_size: int, scale: int
) -> np.ndarray:
    """Fetches a patch of pixels from Earth Engine.

    It retries if we get error "429: Too Many Requests".

    Args:
        image: Image to get the patch from.
        lonlat: A (longitude, latitude) pair for the point of interest.
        patch_size: Size in pixels of the surrounding square patch.
        scale: Number of meters per pixel.

    Raises:
        requests.exceptions.RequestException

    Returns: The requested patch of pixels as a NumPy array with shape (width, height, bands).
    """
    point = ee.Geometry.Point(lonlat)
    url = image.getDownloadURL(
        {
            "region": point.buffer(scale * patch_size / 2, 1).bounds(1),
            "dimensions": [patch_size, patch_size],
            "format": "NPY",
        }
    )

    # If we get "429: Too Many Requests" errors, it's safe to retry the request.
    # The Retry library only works with `google.api_core` exceptions.
    response = requests.get(url)
    if response.status_code == 429:
        raise exceptions.TooManyRequests(response.text)

    # Still raise any other exceptions to make sure we got valid data.
    response.raise_for_status()
    return np.load(io.BytesIO(response.content), allow_pickle=True)
