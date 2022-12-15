# Copyright 2022 Google LLC
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

"""Trains a Fully Convolutional Network to predict precipitation."""

from __future__ import annotations

from glob import glob
import os
from typing import Any as AnyType, Iterable, Optional

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
import numpy as np
import torch
from transformers import PretrainedConfig, PreTrainedModel, Trainer, TrainingArguments


# Default values.
EPOCHS = 100
BATCH_SIZE = 512
TRAIN_TEST_RATIO = 0.9


# https://huggingface.co/docs/transformers/main/en/custom_models#writing-a-custom-configuration
class WeatherConfig(PretrainedConfig):
    model_type = "weather"

    def __init__(
        self,
        mean: list = [],
        std: list = [],
        num_inputs: int = 52,
        num_hidden1: int = 64,
        num_hidden2: int = 128,
        num_outputs: int = 2,
        kernel_size: tuple[int, int] = (3, 3),
        **kwargs: AnyType,
    ) -> None:
        self.mean = mean
        self.std = std
        self.num_inputs = num_inputs
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        super().__init__(**kwargs)


# https://huggingface.co/docs/transformers/main/en/custom_models#writing-a-custom-model
class WeatherModel(PreTrainedModel):
    config_class = WeatherConfig

    def __init__(self, config: WeatherConfig) -> None:
        super().__init__(config)
        self.loss = torch.nn.SmoothL1Loss()
        self.model = torch.nn.Sequential(
            Normalization(config.mean, config.std),
            MoveDim(-1, 1),  # convert to channels-first
            torch.nn.Conv2d(config.num_inputs, config.num_hidden1, config.kernel_size),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                config.num_hidden1, config.num_hidden2, config.kernel_size
            ),
            torch.nn.ReLU(),
            MoveDim(1, -1),  # convert to channels-last
            torch.nn.Linear(config.num_hidden2, config.num_outputs),
            torch.nn.ReLU(),  # precipitation cannot be negative
        )

    def forward(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.model(inputs)
        if labels is None:
            return {"logits": outputs}
        loss = self.loss(outputs, labels)
        return {"loss": loss, "logits": outputs}

    @staticmethod
    def create(inputs: Dataset, **kwargs: AnyType) -> WeatherModel:
        data = np.array(inputs, np.float32)
        mean = data.mean(axis=(0, 1, 2))[None, None, None, :]
        std = data.std(axis=(0, 1, 2))[None, None, None, :]
        config = WeatherConfig(mean.tolist(), std.tolist(), **kwargs)
        return WeatherModel(config)

    def predict(self, inputs: AnyType) -> np.ndarray:
        return self.predict_batch(torch.as_tensor([inputs]))[0]

    def predict_batch(self, inputs_batch: AnyType) -> np.ndarray:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.to(device)
        with torch.no_grad():
            outputs = model(torch.as_tensor(inputs_batch, device=device))
            predictions = outputs["logits"]
            return predictions.cpu().numpy()


class Normalization(torch.nn.Module):
    """Preprocessing normalization layer with z-score."""

    def __init__(self, mean: AnyType, std: AnyType) -> None:
        super().__init__()
        self.mean = torch.nn.Parameter(torch.as_tensor(mean))
        self.std = torch.nn.Parameter(torch.as_tensor(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class MoveDim(torch.nn.Module):
    def __init__(self, src: int, dest: int) -> None:
        super().__init__()
        self.src = src
        self.dest = dest

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.moveaxis(self.src, self.dest)


def create_dataset(data_path: str, train_test_ratio: float) -> DatasetDict:
    def data_iterator() -> Iterable[dict[str, np.ndarray]]:
        for filename in glob(os.path.join(data_path, "*.npz")):
            with open(filename, "rb") as f:
                npz = np.load(f)
                yield {"inputs": npz["inputs"], "labels": npz["labels"]}

    dataset = Dataset.from_generator(data_iterator)
    return dataset.train_test_split(train_size=train_test_ratio, shuffle=True)


def train(
    model: PreTrainedModel,
    dataset: DatasetDict,
    model_path: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> None:
    training_args = TrainingArguments(
        output_dir=os.path.join(model_path, "outputs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(model_path)


def run(
    data_path: str,
    model_path: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    train_test_ratio: float = TRAIN_TEST_RATIO,
) -> None:

    print(f"data_path: {data_path}")
    print(f"model_path: {model_path}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"train_test_ratio: {train_test_ratio}")
    print("-" * 40)

    dataset = create_dataset(data_path, train_test_ratio)
    print(dataset)

    model = WeatherModel.create(dataset["train"]["inputs"])
    print(model.config)
    print(f"mean: {np.array(model.config.mean).shape}")
    print(f"std:  {np.array(model.config.std).shape}")
    print(model)

    train(model, dataset, model_path, epochs, batch_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--train-test-ratio", type=float, default=TRAIN_TEST_RATIO)
    args = parser.parse_args()

    run(**vars(args))
