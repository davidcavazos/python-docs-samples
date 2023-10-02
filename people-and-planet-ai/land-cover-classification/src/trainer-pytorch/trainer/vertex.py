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

from google.cloud import aiplatform

# Default values.
DISPLAY_NAME = "🌍 Land cover"


def run(
    project: str,
    bucket: str,
    location: str,
    data_path: str,
    model_path: str,
    packages: list[str] = [],
    display_name: str = DISPLAY_NAME,
    trainer_args: list[str] = [],
) -> None:
    aiplatform.init(project=project, location=location, staging_bucket=bucket)

    # https://cloud.google.com/vertex-ai/docs/training/pre-built-containers#pytorch
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=display_name,
        python_package_gcs_uri=packages,
        python_module_name="trainer.task",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest",
    )

    job.run(args=[data_path, model_path, *trainer_args])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("model_path")
    parser.add_argument("--project", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--location", default="us-central")
    parser.add_argument("--package", dest="packages", nargs="*")
    parser.add_argument("--display-name", default=DISPLAY_NAME)
    (args, trainer_args) = parser.parse_known_args()

    run(
        project=args.project,
        bucket=args.bucket,
        location=args.location,
        data_path=args.data_path,
        model_path=args.model_path,
        packages=args.packages,
        display_name=args.display_name,
        trainer_args=trainer_args,
    )
