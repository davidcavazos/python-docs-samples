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

from landcover.data import LANDCOVER_NAME
from landcover.data import LANDCOVER_CLASSES


KERNEL_SIZE = 5


def create_model(
    dataset: tf.data.Dataset,
    data_shape: tuple,
    data_types: dict[str, tf.DType],
) -> tf.keras.Model:
    """Creates a Fully Convolutional Network Keras model.

    Make sure you pass the *training* dataset, not the validation or full dataset.

    Args:
        dataset: Training dataset used to normalize inputs.
        kernel_size: Size of the square of neighboring pixels for the model to look at.

    Returns: A compiled fresh new model (not trained).
    """
    # Adapt the preprocessing layers.
    normalization = tf.keras.layers.Normalization(axis=-1)
    dataset_inputs = dataset.map(
        lambda inputs, _: tf.stack(list(inputs.values()), axis=-1)
    )
    normalization.adapt(dataset_inputs)
    print(f"normalization.mean {normalization.mean.shape}:")
    print(normalization.mean.numpy())
    print(f"normalization.variance {normalization.variance.shape}:")
    print(normalization.variance.numpy())

    # Define the Fully Convolutional Network.
    inputs = {
        name: tf.keras.Input((*data_shape, 1), dtype=dtype, name=name)
        for name, dtype in data_types.items()
        if name != LANDCOVER_NAME
    }
    dense_inputs = tf.keras.layers.concatenate(inputs.values())
    num_classes = len(LANDCOVER_CLASSES)
    outputs = tf.keras.Sequential(
        layers=[
            normalization,
            tf.keras.layers.Conv2D(64, KERNEL_SIZE, activation="relu"),
            tf.keras.layers.Conv2DTranspose(32, KERNEL_SIZE, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="fcn",
    )
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs(dense_inputs),
        name="landcover_model",
    )
    print(model.summary())

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.OneHotIoU(
                num_classes=num_classes,
                target_class_ids=list(range(num_classes)),
            )
        ],
    )
    return model
