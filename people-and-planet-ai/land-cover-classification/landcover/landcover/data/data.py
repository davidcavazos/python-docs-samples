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
from google.api_core import retry
import numpy as np

# Constants.
INPUT_HOUR_DELTAS = [-4, -2, 0]
OUTPUT_HOUR_DELTAS = [2, 6]


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


def get_input_image(year: int) -> ee.Image:
    """Gets an Earth Engine image with all the inputs for the model.

    Args:
        year: Year to calculate the median composite.

    Returns: An Earth Engine image with the model inputs.
    """
    sentinel2 = get_sentinel2(year)
    elevation = get_elevation()
    return ee.Image.cat([sentinel2, elevation])


def get_label_image() -> ee.Image:
    """Gets an Earth Engine image with the labels to train the model.

    The labels come from the European Space Agency WorldCover dataset.
    This remaps the ESA classifications with the Dynamic World classifications.
    Missing values are filled with 0, which corresponds to the 'water' classification.

    For more information, see:
        https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100
        https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1

    Returns: An Earth Engine image with the model labels.
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


def get_example_image() -> ee.Image:
    """Gets an Earth Engine image with the labels to train the model.

    The labels come from the European Space Agency WorldCover dataset.
    This remaps the ESA classifications with the Dynamic World classifications.
    Missing values are filled with 0, which corresponds to the 'water' classification.

    For more information, see:
        https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100
        https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1

    Returns: An Earth Engine image with the model labels.
    """
    input_image = get_input_image(2020)

    # Remap the ESA classifications into the Dynamic World classifications
    fromValues = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    toValues = [1, 5, 2, 4, 6, 7, 8, 0, 3, 3, 7]
    label_image = (
        ee.Image("ESA/WorldCover/v100/2020")
        .select("Map")
        .remap(fromValues, toValues)
        .rename("landcover")
        .unmask(0)  # water
        .uint8()
    )
    return ee.Image.cat([input_image, label_image])


@retry.Retry()
def get_patch(
    point: tuple[float, float],
    image: ee.Image,
    patch_size: int,
    crs_code: str,
    crs_scale: tuple[float, float],
) -> np.ndarray:
    """Fetches a patch of pixels from Earth Engine.

    It retries if we get error "429: Too Many Requests".

    Args:
        image: Image to get the patch from.
        point: A (longitude, latitude) pair for the point of interest.
        patch_size: Size in pixels of the surrounding square patch.
        scale: Number of meters per pixel.

    Returns:
        The requested patch of pixels as a structured
        NumPy array with shape (width, height).
    """
    (lon, lat) = point
    (scale_x, scale_y) = crs_scale
    offset_x = -scale_x * (patch_size + 1) / 2
    offset_y = -scale_y * patch_size / 2

    request = {
        "expression": image,
        "fileFormat": "NPY",
        "grid": {
            "dimensions": {"width": patch_size, "height": patch_size},
            "crsCode": crs_code,
            "affineTransform": {
                "scaleX": scale_x,
                "scaleY": scale_y,
                "shearX": 0,
                "shearY": 0,
                "translateX": lon + offset_x,
                "translateY": lat + offset_y,
            },
        },
    }
    return np.load(io.BytesIO(ee.data.computePixels(request)))
