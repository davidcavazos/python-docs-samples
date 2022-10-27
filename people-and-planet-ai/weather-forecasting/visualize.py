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

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_rgb_images(
    values: np.ndarray, min: float = 0.0, max: float = 1.0
) -> np.ndarray:
    """Renders a numeric NumPy array with shape (width, height, rgb) as an image.
    Args:
        values: A float array with shape (width, height, rgb).
        min: Minimum value in the values.
        max: Maximum value in the values.
    Returns: An uint8 array with shape (width, height, rgb).
    """
    scaled_values = (values - min) / (max - min)
    rgb_values = np.clip(scaled_values, 0, 1) * 255
    return rgb_values.astype(np.uint8)


def render_classifications(values: np.ndarray, palette: list[str]) -> np.ndarray:
    """Renders a classifications NumPy array with shape (width, height, 1) as an image.
    Args:
        values: An uint8 array with shape (width, height, 1).
        palette: List of hex encoded colors.
    Returns: An uint8 array with shape (width, height, rgb) with colors from the palette.
    """
    # Create a color map from a hex color palette.
    xs = np.linspace(0, len(palette), 256)
    indices = np.arange(len(palette))

    red = np.interp(xs, indices, [int(c[0:2], 16) for c in palette])
    green = np.interp(xs, indices, [int(c[2:4], 16) for c in palette])
    blue = np.interp(xs, indices, [int(c[4:6], 16) for c in palette])

    color_map = np.array([red, green, blue]).astype(np.uint8).transpose()
    color_indices = (values / len(palette) * 255).astype(np.uint8)
    return np.take(color_map, color_indices, axis=0)


# GOES 16: https://developers.google.com/earth-engine/datasets/catalog/NOAA_GOES_16_MCMIPF
# Elevation: https://developers.google.com/earth-engine/datasets/catalog/MERIT_DEM_v1_0_3
# GPM: https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V06


def render_goes16(patch: np.ndarray) -> np.ndarray:
    red = patch[1]  # CMI_C02
    green = patch[2]  # CMI_C03
    blue = patch[0]  # CMI_C01
    rgb_patch = np.stack([red, green, blue], axis=-1)
    return render_rgb_images(rgb_patch, max=3000)


def render_gpm(patch: np.ndarray) -> np.ndarray:
    palette = [
        "000096",
        "0064ff",
        "00b4ff",
        "33db80",
        "9beb4a",
        "ffeb00",
        "ffb300",
        "ff6400",
        "eb1e00",
        "af0000",
    ]
    return render_classifications(patch[-1], palette)


def show_inputs(data: np.ndarray) -> None:
    patches = data.transpose((1, 0, 2, 3))
    fig = make_subplots(rows=2, cols=len(patches))
    for i, patch in enumerate(patches):
        fig.add_trace(go.Image(z=render_goes16(patch)), row=1, col=i + 1)
        fig.add_trace(go.Image(z=render_gpm(patch)), row=2, col=i + 1)
    fig.show()


def show_labels(data: np.ndarray) -> None:
    patches = data.transpose((1, 0, 2, 3))
    fig = make_subplots(rows=1, cols=len(patches))
    for i, patch in enumerate(patches):
        fig.add_trace(go.Image(z=render_gpm(patch)), row=1, col=i + 1)
    fig.show()
