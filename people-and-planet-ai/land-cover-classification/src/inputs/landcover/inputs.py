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

from collections.abc import Iterator
import io

import ee
from google.api_core import retry
import google.auth
import numpy as np

# Constants.
MAX_ELEVATION = 6000  # found empirically
ELEVATION_BINS = 10

LANDCOVER_NAME = "landcover"
LANDCOVER_CLASSES = {
    "💧 Water": "419BDF",
    "🌳 Trees": "397D49",
    "🌾 Grass": "88B053",
    "🌿 Flooded vegetation": "7A87C6",
    "🚜 Crops": "E49635",
    "🪴 Shrub and scrub": "DFC35A",
    "🏗️ Built-up areas": "C4281B",
    "🪨 Bare ground": "A59B8F",
    "❄️ Snow and ice": "B39FE1",
}

# Polygons covering most land areas in the world.
WORLD_POLYGONS = [
    # Americas
    [(-33.0, -7.0), (-55.0, 53.0), (-166.0, 65.0), (-68.0, -56.0)],
    # Africa, Asia, Europe
    [
        (74.0, 71.0),
        (166.0, 55.0),
        (115.0, -11.0),
        (74.0, -4.0),
        (20.0, -38.0),
        (-29.0, 25.0),
    ],
    # Australia
    [(170.0, -47.0), (179.0, -37.0), (167.0, -12.0), (128.0, 17.0), (106.0, -29.0)],
]


def ee_init() -> None:
    # Get the default credentials to authenticate to Earth Engine.
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


def get_land_cover() -> ee.Image:
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
        .rename(LANDCOVER_NAME)
        .unmask(0)  # water
        .uint8()
    )


def get_input_image(year: int) -> ee.Image:
    """Gets an Earth Engine image with all the inputs for the model.

    Args:
        year: Year to calculate the median composite.

    Returns: An Earth Engine image with the model inputs.
    """
    sentinel2 = get_sentinel2(year)
    elevation = get_elevation()
    return ee.Image.cat([sentinel2, elevation])


def sample_points(
    seed: int, num_samples: int, scale: int
) -> Iterator[tuple[float, float]]:
    """Selects around the same number of points for every classification.

    This expects the input image to be an integer, for balanced regression points
    you could do `image.int()` to truncate the values into an integer.
    If the values are too large, it might be good to bucketize, for example
    the range is between 0 and ~1000 `image.divide(100).int()` would give ~10 buckets.

    Args:
        seed: Random seed to make sure to get different results on different workers.
        num_samples: Total number of points to sample for each bin.
        scale: Number of meters per pixel.

    Yields: Tuples of (longitude, latitude) coordinates.
    """
    land_cover = get_land_cover().select(LANDCOVER_NAME)
    elevation_bins = (
        get_elevation()
        .clamp(0, MAX_ELEVATION)
        .divide(MAX_ELEVATION)
        .multiply(ELEVATION_BINS - 1)
        .uint8()
    )
    num_points = int(0.5 + num_samples / ELEVATION_BINS / len(LANDCOVER_CLASSES))
    unique_bins = elevation_bins.multiply(ELEVATION_BINS).add(land_cover)
    points = unique_bins.stratifiedSample(
        numPoints=max(1, num_points),
        region=ee.Geometry.MultiPolygon(WORLD_POLYGONS),
        scale=scale,
        seed=seed,
        geometries=True,
    )
    for point in points.toList(points.size()).getInfo():
        yield point["geometry"]["coordinates"]


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
    label_image = get_land_cover()
    return ee.Image.cat([input_image, label_image])


@retry.Retry()
def get_patch(
    point: tuple[float, float],
    image: ee.Image,
    patch_size: int,
    crs: str,
    crs_scale: tuple[float, float],
) -> np.ndarray:
    """Fetches a patch of pixels from Earth Engine.

    Args:
        image: Image to get the patch from.
        point: A (longitude, latitude) pair for the point of interest.
        patch_size: Size in pixels of the surrounding square patch.
        crs: Coordinate Reference System code.
        crs_scale: Pair of (scale_x, scale_y) transform for the CRS.

    Returns: A NumPy structured array with shape (width, height).
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
            "crsCode": crs,
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
