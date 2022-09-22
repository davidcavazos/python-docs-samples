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

import conftest
import pytest


@pytest.fixture(scope="session")
def test_name(python_version: str) -> str:
    return f"ppai/weather-forecasting-py{python_version}"


PRELUDE = """
from unittest.mock import Mock
import sys

sys.modules['google.colab'] = Mock()
exit = Mock()

credentials, _ = google.auth.default(
    scopes=[
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/earthengine",
    ]
)
ee.Initialize(
    credentials.with_quota_project(None),
    project='python-docs-samples-tests',
    opt_url="https://earthengine-highvolume.googleapis.com",
)
ee.Initialize = Mock()
"""


def test_notebook(
    project: str, bucket_name: str, location: str, unique_name: str
) -> None:
    # Create the datasets since this step is disabled from testing the notebook.
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

    substitutions = {
        "project": project,
        "bucket": bucket_name,
        "location": location,
        "display_name": unique_name,  # Vertex AI job name
        "epochs": 1,
    }
    conftest.run_notebook("README.ipynb", substitutions, PRELUDE)
