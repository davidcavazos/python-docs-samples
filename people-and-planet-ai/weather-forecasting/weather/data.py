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

# https://developers.google.com/earth-engine/datasets/catalog/NOAA_GOES_16_MCMIPF
# https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V06

from datetime import datetime, timedelta
from typing import Iterable, Tuple

import numpy as np

from .utils import (
    Bounds,
    Example,
    Point,
    ee_init,
    balanced_sample,
    get_image,
    get_sequence,
)


INPUTS = {
    "NOAA/GOES/16/MCMIPF": ["CMI_.*"],
    "NASA/GPM_L3/IMERG_V06": ["precipitationCal"],
}

LABELS = {
    "NASA/GPM_L3/IMERG_V06": ["precipitationCal"],
}

PATCH_SIZE = 32
SCALE = 10000

WINDOW = timedelta(days=1)


def sample_points(date: datetime, bounds: Bounds, num_points: int) -> Iterable[Point]:
    ee_init()
    image = get_image(LABELS, date, WINDOW)
    for lat, lon in balanced_sample(image, bounds, num_points, SCALE):
        yield Point(date, lat, lon)


def get_training_example(point: Point) -> Tuple[Point, Example]:
    ee_init()
    return (point, Example(get_input_data(point), get_label_data(point)))


def get_input_data(point: Point) -> np.ndarray:
    return get_sequence(point, [-2, -1, 0], INPUTS, PATCH_SIZE, SCALE, WINDOW)


def get_label_data(point: Point) -> np.ndarray:
    return get_sequence(point, [1, 6], LABELS, PATCH_SIZE, SCALE, WINDOW)
