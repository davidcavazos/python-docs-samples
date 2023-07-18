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

import apache_beam as beam
import numpy as np
import tensorflow as tf


def serialize_to_tfexample(example: np.ndarray) -> bytes:
    """Serializes an example NumPy array as a tf.Example.

    The NumPy array is a structured array, where each field corresponds to each band.

    Since TensorFlow's TFRecords only support flat lists, we serialize
    our multi-dimensional arrays into a list of bytes.
    When reading it, we must parse it back to its original form.

    Args:
        example: The example array containing inputs and labels.

    Returns: The serialized tf.Example as bytes.
    """
    features = {
        name: tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(example[name]).numpy()]
            )
        )
        for name in example.dtype.names
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


@beam.ptransform_fn
def WriteToTFRecord(
    pcollection: beam.PCollection[np.ndarray], file_path_prefix: str
) -> beam.PCollection[str]:
    return (
        pcollection
        | "TFExample serialize" >> beam.Map(serialize_to_tfexample)
        | "Write TFRecords"
        >> beam.io.WriteToTFRecord(file_path_prefix, file_name_suffix=".tfrecord.gz")
    )
