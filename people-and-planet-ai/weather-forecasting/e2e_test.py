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

from __future__ import annotations

from datetime import datetime
import os
import textwrap

# The conftest contains a bunch of reusable fixtures used all over the place.
# If we use a fixture not defined here, it must be on the conftest!
#   https://docs.pytest.org/en/latest/explanation/fixtures.html
import conftest  # python-docs-samples/people-and-planet-ai/conftest.py

import pytest

from serving import data
from trainer.task import WeatherModel


# ---------- FIXTURES ---------- #


@pytest.fixture(scope="session")
def test_name(python_version: str) -> str:
    # Many fixtures expect a fixture called `test_name`, so be sure to define it!
    return f"ppai/weather-py{python_version}"


@pytest.fixture(scope="session")
def data_path(bucket_name: str) -> str:
    path = "data_local"
    conftest.run_cmd(
        "python",
        "create_dataset.py",
        f"--data-path={path}",
        "--size=2",
        "--num-bins=1",
        "--num-points=1",
    )
    return path


@pytest.fixture(scope="session")
def gcs_data_path(bucket_name: str, data_path: str) -> str:
    gcs_path = f"gs://{bucket_name}/weather/data/"
    conftest.run_cmd("gsutil", "-m", "cp", f"{data_path}/*.npz", gcs_path)
    return gcs_path


@pytest.fixture(scope="session")
def model_local_path() -> str:
    path = "model_local"
    os.makedirs(path)
    conftest.run_cmd("cp", os.path.join("model", "*"), path)
    return path


# @pytest.fixture(scope="session")
# def gcs_model_path(bucket_name: str) -> str:
#     # This is a different path than where Vertex AI saves its model.
#     gcs_path = f"gs://{bucket_name}/model"
#     conftest.run_cmd("gsutil", "-m", "cp", "-r", "./model", gcs_path)
#     return gcs_path


# ---------- TESTS ---------- #


def test_pretrained_model() -> None:
    data.ee_init()
    patch_size = 16
    date = datetime(2019, 9, 3, 18)
    inputs = data.get_inputs_patch(date, (-90.0, 25.0), patch_size)

    model = WeatherModel.from_pretrained("model")
    assert inputs.shape == (patch_size, patch_size, 52)
    predictions = model.predict(inputs)
    assert predictions.shape == (patch_size, patch_size, 2)


def test_weather_forecasting_notebook(
    unique_name: str,
    project: str,
    bucket_name: str,
    location: str,
    data_path: str,
    gcs_data_path: str,
    model_local_path: str,
    # gcs_model_path: str,
) -> None:

    dataflow_dataset_flags = " ".join(
        [
            '--runner="DataflowRunner"',
            f"--job_name={unique_name}-dataset",
            "--num-dates=1",
            "--num-bins=1",
            "--max-requests=1",
        ]
    )

    conftest.run_notebook_parallel(
        "README.ipynb",
        prelude=textwrap.dedent(
            f"""\
            from serving.data import ee_init

            # Google Cloud resources.
            project = {repr(project)}
            bucket = {repr(bucket_name)}
            location = {repr(location)}

            # Initialize Earth Engine.
            ee_init()
            """
        ),
        sections={
            "# 📚 Understand the data": {},
            "# 🗄 Create the dataset": {},
            "# ☁️ Create the dataset in Dataflow": {
                "replace": {'--runner="DataflowRunner"': dataflow_dataset_flags},
            },
            "# 🧠 Train the model": {"variables": {"data_path": data_path}},
            "# ☁️ Train the model in Vertex AI": {
                "variables": {"epochs": 2}
                # gcs_data_path
            },
            "# 🔮 Make predictions": {},  # model_local_path
        },
    )
