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
os.chdir(Path(__file__).parent.parent.parent.parent.parent.resolve())

# ---------- FIXTURES ---------- #


@pytest.fixture(scope="session")
def test_name() -> str:
    return f"ppai/land-cover"


@pytest.fixture(scope="session")
def data_path(bucket_name: str) -> str:
    data_path = f"gs://{bucket_name}/data/trainer/tf"
    if not tf.io.gfile.exists(data_path):
        conftest.run_cmd(
            "python",
            "-m",
            "landcover.dataset.create",
            data_path,
            "--tfrecords",
            "--num-samples=2",
            "--max-requests=1",
            "--max-size=4",
        )
        assert tf.io.gfile.listdir(data_path), "no data files found"
    return data_path


# ---------- TESTS ---------- #


def test_landcover_trainer_local(bucket_name: str, data_path: str):
    model_path = f"gs://{bucket_name}/model/local/tf"
    conftest.run_cmd(
        "python",
        "-m",
        "trainer.task",
        data_path,
        model_path,
        f"--tensorboard=gs://{bucket_name}/logs/tf",
        "--batch-size=2",
    )
    assert tf.io.gfile.listdir(model_path), "no model files found"


def test_landcover_trainer_vertex(
    project: str, bucket_name: str, location: str, data_path: str
):
    packages = [
        # conftest.build_pkg("src/model/tensorflow", "build/tf"),
        conftest.build_pkg("src/trainer/tensorflow", "build/tf"),
    ]
    conftest.run_cmd("gsutil", "-m", "cp", *packages, f"gs://{bucket_name}/build/tf/")

    # Launch the training job in Vertex AI.
    model_path = f"gs://{bucket_name}/model/vertex/tf"
    conftest.run_cmd(
        "python",
        "-m",
        "trainer.vertex",
        data_path,
        model_path,
        f"--tensorboard=gs://{bucket_name}/logs/tf",
        "--batch-size=2",
        f"--project={project}",
        f"--bucket={bucket_name}",
        f"--location={location}",
        *[f"--package=gs://{bucket_name}/{pkg}" for pkg in packages],
    )
    assert tf.io.gfile.listdir(model_path), "no model files found"
