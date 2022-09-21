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

import logging
import os
import platform
import re
import subprocess
from typing import Dict
import uuid

from google.cloud import storage
from nbclient import NotebookClient
import nbformat
import pytest

PYTHON_VERSION = "".join(platform.python_version_tuple()[0:2])

NAME = f"ppai/weather-forecasting-py{PYTHON_VERSION}"

UUID = uuid.uuid4().hex[0:6]
PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = "us-central1"

PRELUDE = """
from unittest.mock import Mock
import sys
from google.cloud import aiplatform

sys.modules['google.colab'] = Mock()
exit = Mock()
aiplatform.CustomTrainingJob.run = Mock()
"""


@pytest.fixture(scope="session")
def bucket_name() -> str:
    # TODO: REMOVE THIS
    yield "dcavazos-lyra"
    return

    storage_client = storage.Client()

    bucket_name = f"{NAME.replace('/', '-')}-{UUID}"
    bucket = storage_client.create_bucket(bucket_name, location=LOCATION)

    logging.info(f"bucket_name: {bucket_name}")
    yield bucket_name

    subprocess.check_call(["gsutil", "-m", "rm", "-rf", f"gs://{bucket_name}/*"])
    bucket.delete(force=True)


def test_notebook(bucket_name: str) -> None:
    # Create the datasets since this step is disabled from testing the notebook.
    subprocess.run(
        [
            "python",
            "datasets.py",
            f"--output-path=gs://{bucket_name}/weather/data",
            "--num-dates=1",
            "--num-points=1",
            "--runner=DataflowRunner",
            # TODO: Make this fail to see if we get a good stack trace
            # f"--project={PROJECT}",
            # f"--region={LOCATION}",
            # f"--temp_location=gs://{bucket_name}/temp",
            # Parameters for testing only, not used in the notebook.
            f"--job_name={NAME.replace('/', '-')}-{UUID}",
        ],
        check=True,
    )

    substitutions = {
        "project": PROJECT,
        "bucket": bucket_name,
        "location": LOCATION,
        "display_name": f"{NAME.replace('/', '-')}-{UUID}",  # Vertex AI job
        "epochs": 1,
    }
    run_notebook("README.ipynb", substitutions, PRELUDE)


# TODO: move this to a conftest.py
def run_notebook(
    ipynb_file: str,
    substitutions: Dict[str, str] = {},
    prelude: str = "",
    remove_shell_commands: bool = True,
) -> None:
    # Regular expression to match and remove shell commands from the notebook.
    #   https://regex101.com/r/auHETK/1
    shell_command_re = re.compile(r"^!(?:[^\n]+\\\n)*(?:[^\n]+)$", re.MULTILINE)

    # Compile regular expressions for variable substitutions.
    #   https://regex101.com/r/AS7pDq/1
    compiled_substitutions = [
        (re.compile(rf"{name}\s*=\s*.+(?<!,)"), f"{name} = {value}")
        for name, value in substitutions.items()
    ]

    nb = nbformat.read(ipynb_file, as_version=4)
    for cell in nb.cells:
        # Only preprocess code cells.
        if cell["cell_type"] != "code":
            continue

        # Remove shell commands.
        if remove_shell_commands:
            cell["source"] = shell_command_re.sub("", cell["source"])

        # Apply variable substitutions.
        for regex, new_value in compiled_substitutions:
            cell["source"] = regex.sub(new_value, cell["source"])

    # Prepend the prelude cell.
    nb.cells = [nbformat.v4.new_code_cell(prelude)] + nb.cells

    # Run the notebook.
    client = NotebookClient(nb)
    client.execute()
