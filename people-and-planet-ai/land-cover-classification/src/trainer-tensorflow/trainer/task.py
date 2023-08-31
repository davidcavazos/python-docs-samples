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

"""Trains a TensorFlow Keras model to classify land cover.

The model is a simple Fully Convolutional Network (FCN).
"""

from __future__ import annotations

import json
import numpy as np
import tensorflow as tf

from landcover.data import LANDCOVER_NAME
from landcover.data import LANDCOVER_CLASSES
from landcover.model.tf_model import create_model

# Default values.
EPOCHS = 10
BATCH_SIZE = 512


def load_dataset(
    filenames: list[str],
    data_shape: tuple,
    data_types: dict[str, tf.DType],
    batch_size: int,
) -> tf.data.Dataset:
    """Reads compressed TFRecord files from a directory into a tf.data.Dataset.

    Args:
        filenames: List of local or Cloud Storage TFRecord files paths.

    Returns: A tf.data.Dataset with the contents of the TFRecord files.
    """

    def with_shape(tensor: tf.Tensor, shape: tuple) -> tf.Tensor:
        tensor.set_shape(shape)
        return tensor

    def deserialize(serialized: tf.Tensor) -> dict[str, tf.Tensor]:
        features = {
            name: tf.io.FixedLenFeature([], tf.string) for name in data_types.keys()
        }
        example = tf.io.parse_example(serialized, features)
        return {
            name: with_shape(tf.io.parse_tensor(example[name], dtype), data_shape)
            for name, dtype in data_types.items()
        }

    def preprocess(
        example: dict[str, tf.Tensor]
    ) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        inputs = example.copy()
        labels_idx = inputs.pop(LANDCOVER_NAME)
        labels = tf.one_hot(labels_idx, len(LANDCOVER_CLASSES))
        return (inputs, labels)

    dataset = (
        tf.data.TFRecordDataset(filenames, compression_type="GZIP")
        .map(deserialize, tf.data.AUTOTUNE)
        .map(preprocess, tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def run(
    data_shape: tuple,
    data_types: dict[str, tf.DType],
    training_files: list[str],
    validation_files: list[str],
    model_path: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tf.keras.Model:
    """Creates and trains the model.

    Args:
        data_path: Local or Cloud Storage directory path where the TFRecord files are.
        model_path: Local or Cloud Storage directory path to store the trained model.
        epochs: Number of times the model goes through the training dataset during training.
        batch_size: Number of examples per training batch.

    Returns: The trained model.
    """
    print(f"{len(training_files)=}")
    print(f"{len(validation_files)=}")
    print(f"{epochs=}")
    print(f"{batch_size=}")
    print("-" * 40)

    # Load the training and validation datasets
    training_dataset = load_dataset(training_files, data_shape, data_types, batch_size)
    validation_dataset = load_dataset(
        validation_files, data_shape, data_types, batch_size
    )

    model = create_model(training_dataset, data_shape, data_types)
    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
    )
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    return model


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamps a value between a minimum and maximum value.

    Args:
        value: The value to clamp.
        min: The minimum value.
        max: The maximum value.

    Returns: The clamped value.
    """
    return min(max(value, min_value), max_value)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        help="Local or Cloud Storage directory for dataset files.",
    )
    parser.add_argument(
        "model_path",
        help="Local or Cloud Storage directory path to store the trained model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of times the model goes through the training dataset during training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of examples per training batch.",
    )
    parser.add_argument(
        "--train-split",
        default=0.9,
        type=lambda x: clamp(float(x), 0, 1),
        help="Percentage of data files to use for training, between 0 and 1.",
    )
    args = parser.parse_args()

    with tf.io.gfile.GFile(tf.io.gfile.join(args.dataset, "schema.json")) as f:
        schema = json.load(f)
        data_shape = tuple(schema["shape"])
        data_types = {
            name: tf.dtypes.as_dtype(dtype) for name, dtype in schema["dtypes"].items()
        }

    # Split the dataset into training and validation subsets.
    filenames = tf.io.gfile.glob(f"{args.dataset}/*.tfrecord.gz")
    split_idx = int(clamp(len(filenames) * args.train_split, 1, len(filenames) - 1))
    training_files = filenames[:split_idx]
    validation_files = filenames[split_idx:]

    run(
        data_shape=data_shape,
        data_types=data_types,
        training_files=training_files,
        validation_files=validation_files,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
