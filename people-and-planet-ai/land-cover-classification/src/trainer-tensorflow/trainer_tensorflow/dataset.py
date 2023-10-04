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

import json
from typing import BinaryIO

import numpy as np
import tensorflow as tf

from trainer_tensorflow.model import NUM_CLASSES

LABEL_NAME = "landcover"
SHUFFLE_BUFFER = 1000


def load_dataset(data_dir: str, schema_file: str = "schema.json") -> tf.data.Dataset:
    """Reads compressed TFRecord files from a directory into a tf.data.Dataset.

    Args:
        filenames: List of local or Cloud Storage TFRecord files paths.

    Returns: A tf.data.Dataset with the contents of the TFRecord files.
    """

    with tf.io.gfile.GFile(tf.io.gfile.join(data_dir, schema_file)) as f:
        (shape, dtypes) = load_schema(f)
    features = {name: tf.io.FixedLenFeature([], tf.string) for name in dtypes.keys()}

    def parse_example(serialized: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        example = tf.io.parse_example(serialized, features)
        inputs = [
            tf.ensure_shape(tf.io.parse_tensor(serialized, dtypes[name]), shape)
            for name, serialized in example.items()
            if name != LABEL_NAME
        ]
        labels = tf.ensure_shape(
            tf.io.parse_tensor(example[LABEL_NAME], dtypes[LABEL_NAME]), shape
        )
        return (tf.stack(inputs, axis=-1), tf.one_hot(labels, NUM_CLASSES, axis=-1))

    file_pattern = tf.io.gfile.join(data_dir, "*.tfrecord.gz")
    data_files = tf.io.gfile.glob(file_pattern)
    dataset = (
        tf.data.TFRecordDataset(
            data_files,
            compression_type="GZIP",
            num_parallel_reads=tf.data.AUTOTUNE,
        )
        .map(parse_example, tf.data.AUTOTUNE)
        .cache()
        .shuffle(SHUFFLE_BUFFER)
    )
    return dataset


def load_schema(fp: BinaryIO) -> tuple[tuple, dict[str, tf.DType]]:
    schema = json.load(fp)
    shape = tuple(schema["shape"])
    dtypes = {
        name: tf.dtypes.as_dtype(dtype) for name, dtype in schema["dtypes"].items()
    }
    return (shape, dtypes)


def serialize(array: np.ndarray) -> bytes:
    fields = {name: tf.convert_to_tensor(array[name]) for name in array.dtype.names}
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                name: tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.io.serialize_tensor(data).numpy()]
                    )
                )
                for name, data in fields.items()
            }
        )
    )
    return example.SerializeToString()
