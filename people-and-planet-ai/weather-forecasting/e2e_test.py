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

from typing import Callable

import conftest
import pytest

PRELUDE = """
from unittest.mock import Mock
import sys

sys.modules['google.colab'] = Mock()
exit = Mock()
"""


@pytest.fixture(scope="session")
def test_name(python_version: str) -> str:
    return f"ppai/weather-forecasting-py{python_version}"


@pytest.fixture(scope="session")
def create_datasets(
    project: str, bucket_name: str, location: str, unique_name: str
) -> None:
    # Since creating the datasets is a shell command, it is disabled
    # in the notebook, so we run it here.
    # ⚠️ If this command changes, please update the notebook!
    conftest.run_cmd(
        "python",
        "datasets.py",
        f"--output-path=gs://{bucket_name}/weather/data",
        "--num-dates=1",
        "--num-points=1",
        "--runner=DataflowRunner",
        f"--project={project}",
        f"--region={location}",
        f"--temp_location=gs://{bucket_name}/temp",
        # Parameters for testing only, not used in the notebook.
        f"--job_name={unique_name}",  # Dataflow job name
    )
    # No need to clean up, files are deleted when the bucket is deleted.


@pytest.fixture(scope="session")
def model_url(
    bucket_name: str,
    cloud_run_deploy: Callable[..., str],
) -> str:
    # Since deploying the model is a shell command, it is disabled
    # in the notebook, so we run it here.
    # ⚠️ If the command flags change, please update the notebook!
    #   https://cloud.google.com/sdk/gcloud/reference/run/deploy
    return cloud_run_deploy(
        "serving",  # source_dir
        f"--update-env-vars=MODEL_PATH=gs://{bucket_name}/weather/model.pt",
        # "--memory=1G",
        "--no-allow-unauthenticated",
    )


def test_notebook(
    project: str,
    bucket_name: str,
    location: str,
    unique_name: str,
    create_datasets: None,
    model_url: str,
) -> None:
    substitutions = {
        "project": project,
        "bucket": bucket_name,
        "location": location,
        "display_name": unique_name,  # Vertex AI job name
        "epochs": 1,
        "model_url": model_url,
    }
    conftest.run_notebook("README.ipynb", substitutions, PRELUDE)
