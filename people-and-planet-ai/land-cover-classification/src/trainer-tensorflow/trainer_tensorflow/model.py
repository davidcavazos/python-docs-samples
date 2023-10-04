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

NUM_CLASSES = 9


def create_model(
    train_dataset: tf.data.Dataset,
    kernel_size: int = 5,
    hidden1: int = 32,
    hidden2: int = 64,
) -> tf.keras.Model:
    """Creates a Fully Convolutional Network Keras model.

    Make sure you pass the *training* dataset, not the validation or full dataset.

    Args:
        dataset: Training dataset used to normalize inputs.
        kernel_size: Size of the square of neighboring pixels for the model to look at.

    Returns: A compiled fresh new model (not trained).
    """
    # Adapt the preprocessing layers.
    normalize = tf.keras.layers.Normalization(axis=-1)
    normalize.adapt(train_dataset.map(lambda x, _: x))
    print(f"{normalize.mean.shape=}")
    print(f"{normalize.mean.numpy()=}")
    print(f"{normalize.variance.shape=}")
    print(f"{normalize.variance.numpy()=}")

    # Define the Fully Convolutional Network.
    num_inputs = normalize.mean.shape[-1]
    layers = [
        tf.keras.Input((None, None, num_inputs), dtype=tf.float32),
        normalize,
        tf.keras.layers.Conv2D(hidden1, kernel_size, activation="relu"),
        tf.keras.layers.Conv2DTranspose(hidden2, kernel_size, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
    model = tf.keras.Sequential(layers)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.OneHotIoU(
                num_classes=NUM_CLASSES,
                target_class_ids=list(range(NUM_CLASSES)),
            )
        ],
    )
    return model
