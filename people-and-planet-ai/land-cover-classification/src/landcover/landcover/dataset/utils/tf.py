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


def serialize_tf(array: np.ndarray) -> bytes:
    fields = (
        {name: tf.convert_to_tensor(array[name]) for name in array.dtype.names}
        if array.dtype.names
        else {"array": tf.convert_to_tensor(array)}
    )
    data = {
        f"{name}.data": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(data).numpy()])
        )
        for name, data in fields.items()
    }
    dtypes = {
        f"{name}.dtype": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[data.dtype.name.encode()])
        )
        for name, data in fields.items()
    }
    example = tf.train.Example(features=tf.train.Features(feature=data | dtypes))
    return example.SerializeToString()


def deserialize_tf(data: bytes) -> dict[str, tf.Tensor]:
    example = tf.train.Example()
    example.ParseFromString(data)
    names = {name.split(".")[0] for name in example.features.feature}
    return {
        name: tf.io.parse_tensor(
            example.features.feature[f"{name}.data"].bytes_list.value[0],
            out_type=tf.dtypes.as_dtype(
                example.features.feature[f"{name}.dtype"].bytes_list.value[0].decode()
            ),
        )
        for name in names
    }
