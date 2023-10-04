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


class DeserlializeInput(tf.keras.layers.Layer):
    def call(self, tensor):
        return_dict = {}

        for k, v in tensor.items():
            decoded = tf.io.decode_base64(v)
            return_dict[k] = tf.map_fn(
                lambda x: tf.io.parse_tensor(x, tf.float32),
                decoded,
                fn_output_signature=tf.float32,
            )

        return return_dict


class ReserlializeOutput(tf.keras.layers.Layer):
    def call(self, tensor_input):
        return tf.map_fn(
            lambda x: tf.io.encode_base64(tf.io.serialize_tensor(x)),
            tensor_input,
            fn_output_signature=tf.string,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_dir, compile=False)

    inputs = {
        model.inputs[0].name: tf.keras.Input(
            shape=[], dtype=tf.string, name="array_image"
        )
    }

    input_deserializer = DeserlializeInput()
    output_deserilaizer = ReserlializeOutput()

    updated_model_input = input_deserializer(serlialized_inputs)
    updated_model = model(updated_model_input)
    updated_model = output_deserilaizer(updated_model)
    updated_model = tf.keras.Model(serlialized_inputs, updated_model)
