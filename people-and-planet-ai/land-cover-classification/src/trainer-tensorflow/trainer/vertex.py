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


def run(project: str, bucket: str, location: str) -> None:
    aiplatform.init(project=project, location=location, staging_bucket=bucket)

    # Ignore the next two lines of code if the experiment you are using already
    # has backing tensorboard instance.
    # tb_instance = aiplatform.Tensorboard.create()
    # aiplatform.init(experiment=experiment, experiment_tensorboard=tb_instance)

    packages = [
        f"gs://{bucket}/landcover/landcover-1.0.0.tar.gz",
        f"gs://{bucket}/landcover/trainer-1.0.0.tar.gz",
    ]

    # https://cloud.google.com/vertex-ai/docs/training/pre-built-containers
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name="🌍 Land cover",
        python_package_gcs_uri=packages,
        python_module_name="trainer.task",
        container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest",
    )

    job.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=str)
    parser.add_argument("--bucket", required=True, type=str)
    parser.add_argument("--location", type=str, default="us-central")
    args = parser.parse_args()

    run(
        project=args.project,
        bucket=args.bucket,
        location=args.location,
    )
