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

import tensorflow as tf
import numpy as np


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


def deserialize(
    serialized: tf.Tensor, schema: dict[str, tf.DType]
) -> dict[str, tf.Tensor]:
    features = {name: tf.io.FixedLenFeature([], tf.string) for name in schema.keys()}
    example = tf.io.parse_example(serialized, features)
    return {
        name: tf.io.parse_tensor(example[name], dtype) for name, dtype in schema.items()
    }
