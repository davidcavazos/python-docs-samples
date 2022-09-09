from typing import List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_rgb_images(values: np.ndarray, min=0.0, max=1.0) -> np.ndarray:
    scaled_values = (values - min) / (max - min)
    rgb_values = scaled_values * 255
    return rgb_values.astype(np.uint8)


def render_classifications(values: np.ndarray, palette: List[str]) -> np.ndarray:
    # Create a color map from a hex color palette.
    xs = np.linspace(0, len(palette), 256)
    indices = np.arange(len(palette))
    color_map = (
        np.array(
            [
                np.interp(xs, indices, [int(c[0:2], 16) for c in palette]),  # red
                np.interp(xs, indices, [int(c[2:4], 16) for c in palette]),  # green
                np.interp(xs, indices, [int(c[4:6], 16) for c in palette]),  # blue
            ]
        )
        .astype(np.uint8)
        .transpose()
    )

    color_indices = (values / len(palette) * 255).astype(np.uint8)
    return np.take(color_map, color_indices, axis=0)


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


def show_inputs(data: np.ndarray):
    patches = data.transpose((1, 0, 2, 3))
    fig = make_subplots(rows=2, cols=len(patches))
    for i, patch in enumerate(patches):
        fig.add_trace(go.Image(z=render_goes16(patch)), row=1, col=i + 1)
        fig.add_trace(go.Image(z=render_gpm(patch)), row=2, col=i + 1)
    fig.show()


def show_labels(data: np.ndarray):
    patches = data.transpose((1, 0, 2, 3))
    fig = make_subplots(rows=1, cols=len(patches))
    for i, patch in enumerate(patches):
        fig.add_trace(go.Image(z=render_gpm(patch)), row=1, col=i + 1)
    fig.show()
