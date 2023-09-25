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

import tensorflow as tf

from landcover.inputs import LANDCOVER_NAME
from landcover.inputs import LANDCOVER_CLASSES
from landcover.model import create_model
from landcover.model import deserialize
from landcover.model import load_schema

# Default values.
EPOCHS = 10
BATCH_SIZE = 4096


def load_dataset(
    filenames: list[str],
    shape: tuple,
    dtypes: dict[str, tf.DType],
    batch_size: int,
) -> tf.data.Dataset:
    """Reads compressed TFRecord files from a directory into a tf.data.Dataset.

    Args:
        filenames: List of local or Cloud Storage TFRecord files paths.

    Returns: A tf.data.Dataset with the contents of the TFRecord files.
    """

    def parse_example(serialized: tf.Tensor) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        inputs = deserialize(serialized, shape, dtypes)
        labels = inputs.pop(LANDCOVER_NAME)
        return (inputs, tf.one_hot(labels, len(LANDCOVER_CLASSES)))

    dataset = (
        tf.data.TFRecordDataset(filenames, compression_type="GZIP")
        .map(parse_example, tf.data.AUTOTUNE)
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def run(
    shape: tuple,
    dtypes: dict[str, tf.DType],
    filenames: list[str],
    train_split: float,
    model_path: str,
    tensorboard_path: str,
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
    print(f"{epochs=}")
    print(f"{batch_size=}")

    # Load the training and validation datasets
    dataset = load_dataset(filenames, shape, dtypes, batch_size)
    total_batches = sum(1 for _ in dataset)
    training_batches = int(total_batches * train_split)

    profile_batch_start = int(training_batches * 0.2)
    profile_batch_end = min(profile_batch_start + 10, training_batches - 1)
    profile_batches = (profile_batch_start, profile_batch_end)

    print(f"{total_batches=}")
    print(f"{training_batches=}")
    print(f"{profile_batches=}")
    print("-" * 40)

    training_dataset = dataset.take(training_batches)
    validation_dataset = dataset.skip(training_batches)
    model = create_model(training_dataset, shape, dtypes)
    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                tensorboard_path,
                histogram_freq=1,
                profile_batch=profile_batches,
                write_images=True,
            )
        ]
        if tensorboard_path
        else [],
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
    parser.add_argument("--tensorboard")
    parser.add_argument("--schema-filename", default="schema.json")
    args = parser.parse_args()

    with tf.io.gfile.GFile(tf.io.gfile.join(args.dataset, args.schema_filename)) as f:
        (shape, dtypes) = load_schema(f)

    run(
        shape=shape,
        dtypes=dtypes,
        filenames=tf.io.gfile.glob(f"{args.dataset}/*.tfrecord.gz"),
        train_split=args.train_split,
        model_path=args.model_path,
        tensorboard_path=args.tensorboard,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
