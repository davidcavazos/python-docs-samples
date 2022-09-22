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
import subprocess
import uuid
from typing import Dict

import nbclient
import nbformat
from google.cloud import storage
import pytest


@pytest.fixture(scope="session")
def project() -> str:
    return os.environ["GOOGLE_CLOUD_PROJECT"]


@pytest.fixture(scope="session")
def location() -> str:
    return "us-central1"


@pytest.fixture(scope="session")
def python_version() -> str:
    return "".join(platform.python_version_tuple()[0:2])


@pytest.fixture(scope="session")
def unique_id() -> str:
    return uuid.uuid4().hex[0:6]


@pytest.fixture(scope="session")
def unique_name(test_name: str, unique_id: str) -> str:
    return f"{test_name.replace('/', '-')}-{unique_id}"


@pytest.fixture(scope="session")
def bucket_name(test_name: str, location: str, unique_id: str) -> str:
    storage_client = storage.Client()

    bucket_name = f"{test_name.replace('/', '-')}-{unique_id}"
    bucket = storage_client.create_bucket(bucket_name, location=location)

    logging.info(f"bucket_name: {bucket_name}")
    yield bucket_name

    subprocess.check_call(["gsutil", "-m", "rm", "-rf", f"gs://{bucket_name}/*"])
    bucket.delete(force=True)


def run_cmd(*cmd: str) -> subprocess.CompletedProcess:
    try:
        logging.info(f">> {cmd}")
        p = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logging.info(p.stdout.decode("utf-8"))
        return p
    except subprocess.CalledProcessError as e:
        # Include the error message from the failed command.
        logging.info(e.stdout.decode("utf-8"))
        raise Exception(f"{e}\n\n{e.stderr.decode('utf-8')}") from e


def run_notebook(
    ipynb_file: str,
    substitutions: dict = {},
    prelude: str = "",
    remove_shell_commands: bool = True,
) -> None:
    # Regular expression to match and remove shell commands from the notebook.
    #   https://regex101.com/r/auHETK/1
    shell_command_re = re.compile(r"^!(?:[^\n]+\\\n)*(?:[^\n]+)$", re.MULTILINE)

    # Compile regular expressions for variable substitutions.
    #   https://regex101.com/r/AS7pDq/1
    compiled_substitutions = [
        (re.compile(rf"{name}\s*=\s*.+(?<!,)"), f"{name} = {repr(value)}")
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
    error = ""
    client = nbclient.NotebookClient(nb)
    try:
        client.execute()
    except nbclient.exceptions.CellExecutionError as e:
        # Remove colors and other escape characters to make it easier to read in the logs.
        #   https://stackoverflow.com/a/33925425
        error = re.sub(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]", "", str(e))

    if error:
        raise RuntimeError(error)
