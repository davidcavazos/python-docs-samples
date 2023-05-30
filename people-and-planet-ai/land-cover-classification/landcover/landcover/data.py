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

"""Data utilities to grab data from Earth Engine.
Meant to be used for both training and prediction so the model is
trained on exactly the same data that will be used for predictions.
"""

from __future__ import annotations

import io

import ee
from google.api_core import exceptions, retry
import google.auth
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import requests

# Constants.
INPUT_HOUR_DELTAS = [-4, -2, 0]
OUTPUT_HOUR_DELTAS = [2, 6]


# Authenticate and initialize Earth Engine with the default credentials.
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


def get_sentinel2(year: int, default_value: float = 1000.0) -> ee.Image:
    """Gets a Sentinel-2 image for the selected date.

    This filters clouds and returns the median for the selected time range.
    Then it removes the mask and fills all the missing values, otherwise
    the data normalization will give infinities and not-a-number.

    Missing values are filled with 1000, which is near the mean.

    For more information:
        https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED

    Args:
        year: Year to calculate the median composite.

    Returns: An Earth Engine image.
    """

    def mask_sentinel2_clouds(image: ee.Image) -> ee.Image:
        CLOUD_BIT = 10
        CIRRUS_CLOUD_BIT = 11
        bit_mask = (1 << CLOUD_BIT) | (1 << CIRRUS_CLOUD_BIT)
        mask = image.select("QA60").bitwiseAnd(bit_mask).eq(0)
        return image.updateMask(mask)

    return (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate(f"{year}-1-1", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_sentinel2_clouds)
        .select("B.*")  # supports regular expressions
        .median()
        .unmask(default_value)
        .float()
    )


def get_elevation() -> ee.Image:
    """Gets a digital elevation map.

    Missing values are filled with 0, which corresponds with sea level.

    For more information:
        https://developers.google.com/earth-engine/datasets/catalog/MERIT_DEM_v1_0_3

    Returns: An Earth Engine image.
    """
    return ee.Image("MERIT/DEM/v1_0_3").rename("elevation").unmask(0).float()


def get_land_cover_2020() -> ee.Image:
    """Get the European Space Agency WorldCover image.

    This remaps the ESA classifications with the Dynamic World classifications.

    Missing values are filled with 0, which corresponds to the 'water' classification.

    For more information, see:
        https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100
        https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1

    Returns: An Earth Engine image with land cover classification as indices.
    """
    # Remap the ESA classifications into the Dynamic World classifications
    fromValues = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    toValues = [1, 5, 2, 4, 6, 7, 8, 0, 3, 3, 7]
    return (
        ee.Image("ESA/WorldCover/v100/2020")
        .select("Map")
        .remap(fromValues, toValues)
        .rename("landcover")
        .unmask(0)  # water
        .uint8()
    )


def get_input_image(year: int) -> ee.Image:
    """Gets an Earth Engine image with all the inputs for the model.

    Args:
        year: Year to calculate the median composite.

    Returns: An Earth Engine image.
    """
    sentinel2 = get_sentinel2(year)
    elevation = get_elevation()
    return ee.Image([sentinel2, elevation])


def get_label_image() -> ee.Image:
    """Gets an Earth Engine image with the labels to train the model.

    Args:
        date: Date to take a snapshot from.

    Returns: An Earth Engine image.
    """
    return get_land_cover_2020()


@retry.Retry()
def get_patch(image: ee.Image, point: tuple, patch_size: int, scale: int) -> np.ndarray:
    """Fetches a patch of pixels from Earth Engine.

    It retries if we get error "429: Too Many Requests".

    Args:
        image: Image to get the patch from.
        point: A (longitude, latitude) pair for the point of interest.
        patch_size: Size in pixels of the surrounding square patch.
        scale: Number of meters per pixel.

    Raises:
        requests.exceptions.RequestException

    Returns:
        The requested patch of pixels as a structured
        NumPy array with shape (width, height).
    """
    geometry = ee.Geometry.Point(point)
    url = image.getDownloadURL(
        {
            "region": geometry.buffer(scale * patch_size / 2, 1).bounds(1),
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
    struct_array = np.load(io.BytesIO(response.content), allow_pickle=True)
    return structured_to_unstructured(struct_array)
