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
from typing import Callable

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


@pytest.fixture(scope="session")
def container_image_build(
    project: str, test_name: str, unique_id: str
) -> Callable[[str], str]:
    # https://docs.pytest.org/en/latest/fixture.html#factories-as-fixtures
    built_images = []

    def build(source_dir: str = ".") -> str:
        image_name = (
            f"gcr.io/{project}/{test_name}/{source_dir}:{unique_id}"
            if source_dir != "."
            else f"gcr.io/{project}/{test_name}:{unique_id}"
        )

        # https://cloud.google.com/sdk/gcloud/reference/builds/submit
        run_cmd(
            "gcloud",
            "builds",
            "submit",
            source_dir,
            f"--project={project}",
            f"--pack=image={image_name}",
            "--quiet",
        )
        built_images.append(image_name)
        logging.info(f"Container image built: {image_name}")
        return image_name

    yield build

    for image_name in built_images:
        # https://cloud.google.com/sdk/gcloud/reference/container/images/delete
        run_cmd(
            "gcloud",
            "container",
            "images",
            "delete",
            image_name,
            f"--project={project}",
            "--force-delete-tags",
            "--quiet",
        )
        logging.info(f"Container image deleted: {image_name}")


@pytest.fixture(scope="session")
def cloud_run_deploy(
    project: str,
    location: str,
    test_name: str,
    unique_id: str,
    container_image_build: Callable[[str], str],
) -> Callable[..., str]:
    # https://docs.pytest.org/en/latest/fixture.html#factories-as-fixtures
    deployed_services = []

    def deploy(source_dir: str, *flags: str) -> str:
        container_image = container_image_build(source_dir)
        service_name = (
            f"{test_name}-{source_dir}-{unique_id}"
            if source_dir != "."
            else f"{test_name}-{unique_id}"
        ).replace("/", "-")

        # https://cloud.google.com/sdk/gcloud/reference/run/deploy
        run_cmd(
            "gcloud",
            "run",
            "deploy",
            service_name,
            f"--project={project}",
            f"--region={location}",
            f"--image={container_image}",
            *flags,
        )
        deployed_services.append(service_name)
        logging.info(f"Cloud Run service deployed: {service_name}")

        # https://cloud.google.com/sdk/gcloud/reference/run/services/describe
        service_url = (
            run_cmd(
                "gcloud",
                "run",
                "services",
                "describe",
                cloud_run_deploy,
                f"--project={project}",
                f"--region={location}",
                "--format=get(status.url)",
            )
            .stdout.decode("utf-8")
            .strip()
        )
        logging.info(f"Cloud Run {service_name} URL: {service_url}")
        return service_url

    yield deploy

    for service_name in deployed_services:
        # https://cloud.google.com/sdk/gcloud/reference/run/services/delete
        run_cmd(
            "gcloud",
            "run",
            "services",
            "delete",
            service_name,
            f"--project={project}",
            f"--region={location}",
            "--quiet",
        )


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
    #   https://regex101.com/r/e32vfW/1
    compiled_substitutions = [
        (
            re.compile(rf"""\b{name}\s*=\s*(?:f?'[^']*'|f?"[^"]*"|\w+)"""),
            f"{name} = {repr(value)}",
        )
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
