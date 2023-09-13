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

from collections.abc import Iterator
import json

import apache_beam as beam
from apache_beam.io.filebasedsink import FileBasedSink
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.filesystems import FileSystems
import numpy as np


@beam.ptransform_fn
def ReadFromNumPy(
    pcoll: beam.pvalue.PBegin, patterns: list[str]
) -> beam.PCollection[str]:
    def read_npz(filename: str) -> Iterator[np.ndarray]:
        with FileSystems.open(filename, "rb") as f:
            data = np.load(f)
            for key in data.files:
                yield data[key]

    return (
        pcoll
        | "Match files" >> beam.Create(FileSystems.match(patterns))
        | "File names" >> beam.FlatMap(lambda m: [f.path for f in m.metadata_list])
        | "Reshuffle" >> beam.Reshuffle()
        | "Read npz" >> beam.FlatMap(read_npz)
    )


class NumPySink(FileBasedSink):
    """A sink to GCS or local NumPy compressed files.

    Init:
        file_path_prefix: The file path to write to. The files written will begin
            with this prefix, followed by a shard identifier (see num_shards), and
            end in a common extension, if given by file_name_suffix. In most cases,
            only this argument is specified and num_shards, shard_name_template, and
            file_name_suffix use default values.
        num_shards: The number of files (shards) used for output. If not set, the
            service will decide on the optimal number of shards.
            Constraining the number of shards is likely to reduce
            the performance of a pipeline.  Setting this value is not recommended
            unless you require a specific number of output files.
    """

    buffer: list[np.ndarray]

    def __init__(self, file_path_prefix: str, num_shards: int = 0):
        super().__init__(
            file_path_prefix,
            file_name_suffix=".npz",
            num_shards=num_shards,
            compression_type=CompressionTypes.UNCOMPRESSED,
            coder=None,
        )

    def open(self, file_handle):
        self.buffer = []
        return super().open(file_handle)

    def write_record(self, file_handle, value):
        self.buffer.append(value)

    def close(self, file_handle):
        arrays_dict = {str(i): arr for i, arr in enumerate(self.buffer)}
        np.savez_compressed(file_handle, **arrays_dict)
        return super().close(file_handle)


@beam.ptransform_fn
def WriteToNumPy(
    pcoll: beam.PCollection[np.ndarray], file_path_prefix: str, num_shards: int = 0
) -> beam.PCollection[str]:
    return pcoll | beam.io.Write(NumPySink(file_path_prefix, num_shards))


@beam.ptransform_fn
def WriteSchema(pcoll: beam.PCollection[str], output_path: str, filename="schema.json"):
    def write_schema(array: np.ndarray) -> None:
        output_file = FileSystems.join(output_path, filename)
        schema = {
            "shape": array.shape,
            "dtypes": {name: array.dtype[name].name for name in array.dtype.names},
        }
        with FileSystems.create(output_file) as f:
            f.write(json.dumps(schema, indent=2).encode("utf-8"))

    return (
        pcoll
        | beam.combiners.Sample.FixedSizeGlobally(1)
        | beam.FlatMap(lambda xs: xs)
        | beam.Map(write_schema)
    )
