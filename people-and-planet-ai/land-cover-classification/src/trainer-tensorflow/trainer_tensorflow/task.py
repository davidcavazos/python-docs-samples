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
from trainer_tensorflow.dataset import load_dataset
from trainer_tensorflow.model import create_model

# Default values.
EPOCHS = 100
BATCH_SIZE = 512
TRAIN_VAL_SPLIT = 0.9


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        help="Local or Cloud Storage directory for dataset files.",
    )
    parser.add_argument(
        "model_dir",
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
        type=float,
        help="Percentage of data files to use for training, between 0 and 1.",
    )
    parser.add_argument("--tensorboard")
    parser.add_argument("--schema-file", default="schema.json")
    args = parser.parse_args()

    assert (
        0 < args.train_split < 1
    ), f"--train-split must be between 0 and 1, got {args.train_split}"
    splits = [args.train_split, 1 - args.train_split]

    gpus = tf.config.list_physical_devices("GPU")
    print(f"{gpus=}")

    dataset = load_dataset(args.data_dir, args.schema_file)
    dataset_size = sum(1 for _ in dataset)
    print(f"{dataset_size=}")
    train_size = int(args.train_split * dataset_size)
    print(f"{train_size=}")
    train_dataset = (
        dataset.take(train_size)
        .batch(args.batch_size)
        .cache()
        .shuffle(100)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        dataset.skip(train_size)
        .batch(args.batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    model = create_model(train_dataset)
    model.summary()

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
    )
    model.save(args.model_dir)
