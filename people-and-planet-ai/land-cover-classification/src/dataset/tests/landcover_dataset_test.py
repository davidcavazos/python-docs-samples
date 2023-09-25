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

from __future__ import annotations

import os
from pathlib import Path


# The conftest contains a bunch of reusable fixtures used all over the place.
# If we use a fixture not defined here, it must be in the conftest!
#   https://docs.pytest.org/en/latest/explanation/fixtures.html
import conftest  # python-docs-samples/people-and-planet-ai/conftest.py
import pytest
import tensorflow as tf

# Change directory to the sample's root directory.
os.chdir(Path(__file__).parent.parent.parent.parent.resolve())

# ---------- FIXTURES ---------- #


@pytest.fixture(scope="session")
def test_name() -> str:
    return f"ppai/land-cover"


# ---------- TESTS ---------- #


def test_landcover_dataset_numpy(
    project: str,
    bucket_name: str,
    location: str,
    unique_name: str,
):
    data_path = f"gs://{bucket_name}/data/np"
    packages = [
        conftest.build_pkg("src/dataset", "build/np"),
        conftest.build_pkg("src/inputs", "build/np"),
    ]
    conftest.run_cmd(
        "python",
        "-m",
        "landcover.dataset.create",
        data_path,
        "--num-samples=4",
        "--max-requests=1",
        "--max-size=8",
        "--runner=DataflowRunner",
        f"--job_name={unique_name}-numpy-dataset",
        f"--project={project}",
        f"--temp_location=gs://{bucket_name}/temp",
        f"--region={location}",
        "--experiments=use_sibling_sdk_workers",
        *[f"--extra_package={pkg}" for pkg in packages],
    )
    assert tf.io.gfile.listdir(data_path), "no data files found"


def test_landcover_dataset_tfrecords(
    project: str,
    bucket_name: str,
    location: str,
    unique_name: str,
):
    data_path = f"gs://{bucket_name}/data/tf"
    packages = [
        conftest.build_pkg("src/dataset", "build/tf"),
        conftest.build_pkg("src/inputs", "build/tf"),
        conftest.build_pkg("src/model/tensorflow", "build/tf"),
    ]
    conftest.run_cmd(
        "python",
        "-m",
        "landcover.dataset.create",
        data_path,
        "--tfrecords",
        "--num-samples=4",
        "--max-requests=1",
        "--max-size=8",
        "--runner=DataflowRunner",
        f"--job_name={unique_name}-tfrecords-dataset",
        f"--project={project}",
        f"--temp_location=gs://{bucket_name}/temp",
        f"--region={location}",
        "--experiments=use_sibling_sdk_workers",
        *[f"--extra_package={pkg}" for pkg in packages],
    )
    assert tf.io.gfile.listdir(data_path), "no data files found"
