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


class DecodeInputs(tf.keras.layers.Layer):
    def call(self, batch: tf.Tensor, dtype: tf.DType = tf.float32) -> tf.Tensor:
        return tf.map_fn(
            lambda x: tf.io.parse_tensor(x, dtype),
            tf.io.decode_base64(batch),
            fn_output_signature=dtype,
        )


class EncodeOutputs(tf.keras.layers.Layer):
    def call(self, batch: tf.Tensor) -> tf.Tensor:
        return tf.map_fn(
            lambda x: tf.io.encode_base64(tf.io.serialize_tensor(x)),
            batch,
            fn_output_signature=tf.string,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("model_ee_dir")
    parser.add_argument("--input-name", default="array")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_dir)

    inputs = {
        args.input_name: tf.keras.Input(shape=[], dtype=tf.string, name=args.input_name)
    }
    outputs = tf.keras.Sequential([DecodeInputs(), model, EncodeOutputs()])
    model_ee = tf.keras.Model(inputs, outputs(inputs[args.input_name]))
    model_ee.summary()

    model_ee.save(args.model_ee_dir)
