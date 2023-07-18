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


class NumPySink(FileBasedSink):
    def __init__(self, file_path_prefix: str) -> None:
        super().__init__(file_path_prefix, coder=None, file_name_suffix=".npz")
        self.examples = []

    def write_record(self, file_handle: BinaryIO, example: np.ndarray):
        self.examples.append(example)

    def close(self, file_handle: BinaryIO):
        batch = np.stack(self.examples)
        values = {name: batch[name] for name in batch.dtype.names}
        np.savez_compressed(file_handle, **values)
        return super().close(file_handle)


@beam.ptransform_fn
def WriteToNumPy(
    pcollection: beam.PCollection[np.ndarray], file_path_prefix: str
) -> beam.PCollection[str]:
    return pcollection | beam.io.Write(NumPySink(file_path_prefix))
