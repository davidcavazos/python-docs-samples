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

import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.filesystems import FileSystems

from landcover.dataset.utils.tf import serialize_tf
from landcover.dataset.utils.beam_np import ReadFromNumPy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        help="Directory path for input NumPy files.",
    )
    parser.add_argument(
        "output_path",
        help="Directory path for output TFRecord files.",
    )
    args, beam_args = parser.parse_known_args()

    logging.getLogger().setLevel(logging.INFO)

    input_pattern = FileSystems.join(args.input_path, "*.npz")
    output_path = FileSystems.join(args.output_path, "examples")
    beam_options = PipelineOptions(beam_args, pickle_library="cloudpickle")
    with beam.Pipeline(options=beam_options) as pipeline:
        dataset = pipeline | "📖 Read NumPy" >> ReadFromNumPy([input_pattern])

        _ = (
            dataset
            | "✍️ To tf.Example" >> beam.Map(serialize_tf)
            | "📝 Write to TFRecord"
            >> beam.io.WriteToTFRecord(
                file_path_prefix=output_path,
                file_name_suffix=".tfrecord.gz",
            )
        )
