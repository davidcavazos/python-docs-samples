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

from typing import BinaryIO

import apache_beam as beam
from apache_beam.io.filebasedsink import FileBasedSink
import numpy as np


class SafeTensorsSink(FileBasedSink):
    def __init__(self, file_path_prefix: str, tensor_format: str) -> None:
        super().__init__(
            file_path_prefix,
            coder=None,
            file_name_suffix=f".{tensor_format}.safetensors",
        )
        self.tensor_format = tensor_format
        self.examples = []

    def write_record(self, file_handle: BinaryIO, example: np.ndarray):
        self.examples.append(example)

    def close(self, file_handle: BinaryIO):
        batch = np.stack(self.examples)
        match self.tensor_format:
            case "numpy":
                from safetensors.numpy import save

                values = {name: batch[name] for name in batch.dtype.names}
                byte_data = save(values)

            case "tensorflow":
                import tensorflow as tf
                from safetensors.tensorflow import save

                values = {
                    name: tf.convert_to_tensor(batch[name])
                    for name in batch.dtype.names
                }
                byte_data = save(values)

            case "torch":
                import torch
                from safetensors.torch import save

                values = {
                    name: torch.from_numpy(batch[name]) for name in batch.dtype.names
                }
                byte_data = save(values)

            case tensor_format:
                raise ValueError(f"Tensor format not supported: {tensor_format}")

        file_handle.write(byte_data)
        return super().close(file_handle)


@beam.ptransform_fn
def WriteToSafeTensors(
    pcollection: beam.PCollection[np.ndarray], file_path_prefix: str, tensor_format: str
) -> beam.PCollection[str]:
    return pcollection | beam.io.Write(SafeTensorsSink(file_path_prefix, tensor_format))
