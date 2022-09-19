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

from glob import glob
from typing import Optional, Tuple, BinaryIO
import os

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, random_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WeatherDataset(Dataset):
    def __init__(self, data_path: str):
        self.files = glob(os.path.join(data_path, "**", "*.npz"), recursive=True)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with open(self.files[idx], "rb") as f:
            npz = np.load(f)
            (inputs, labels) = (npz["inputs"], npz["labels"])
            return (torch.from_numpy(inputs), torch.from_numpy(labels))


class Normalize(torch.nn.Module):
    """Preprocessing normalization layer."""

    def __init__(self, std: torch.Tensor, mean: torch.Tensor):
        super().__init__()
        self.std = std.to(DEVICE)
        self.mean = mean.to(DEVICE)

    def forward(self, x):
        return (x - self.mean) / self.std


class Model(torch.nn.Module):
    def __init__(
        self,
        std: torch.Tensor,
        mean: torch.Tensor,
        kernel_size: int,
        hidden_units: int = 8,
    ):
        super(Model, self).__init__()
        self.layers = torch.nn.Sequential(
            Normalize(std, mean),
            torch.nn.Conv3d(
                in_channels=17,
                out_channels=hidden_units,
                kernel_size=(3, kernel_size, kernel_size),
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(
                in_channels=hidden_units,
                out_channels=1,
                kernel_size=(2, kernel_size, kernel_size),
            ),
        )
        self.loss = torch.nn.SmoothL1Loss()

    def forward(self, x: torch.Tensor):
        return self.layers(x)


def std_mean(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    num_channels = dataset[0][0].shape[0]
    total_sum = torch.zeros([num_channels])
    squared_sum = torch.zeros([num_channels])
    for inputs, _ in dataset:
        dimensions = list(range(1, len(inputs.shape)))
        total_sum += torch.mean(inputs, dimensions)
        squared_sum += torch.mean(inputs**2, dimensions)
    mean = total_sum / len(dataset)
    std = (squared_sum / len(dataset) - mean**2) ** 0.5

    shape = [num_channels] + [1] * len(dataset[0][0].shape[1:])
    return (std.reshape(shape), mean.reshape(shape))


def split_dataset(dataset: Dataset, train_test_ratio: float) -> Tuple[Dataset, Dataset]:
    train_size = int(len(dataset) * train_test_ratio)
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def train(
    model: Model,
    dataset: Dataset,
    optimizer: Optimizer,
    batch_size: int = 8,
) -> float:
    dataloader = DataLoader(
        dataset, batch_size, num_workers=os.cpu_count(), shuffle=True
    )
    total_loss = 0.0

    model.train()
    for inputs, labels in dataloader:
        (inputs, labels) = (inputs.to(DEVICE), labels.to(DEVICE))

        # Compute prediction error
        predictions = model(inputs)
        loss = model.loss(predictions, labels)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)


def test(model: Model, dataset: Dataset, batch_size: int = 8) -> float:
    dataloader = DataLoader(dataset, batch_size, num_workers=os.cpu_count())
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            (inputs, labels) = (inputs.to(DEVICE), labels.to(DEVICE))

            predictions = model(inputs)
            total_loss += model.loss(predictions, labels).item()
    return total_loss / len(dataloader)


def fit(
    model: Model,
    train_dataset: Dataset,
    test_dataset: Dataset,
    epochs: int = 100,
    batch_size: int = 8,
    optimizer: Optional[Optimizer] = None,
) -> Model:
    optimizer = optimizer or torch.optim.Adam(model.parameters())
    print(f"Optimizer: {optimizer}")

    for epoch in range(epochs):
        train_loss = train(model, train_dataset, optimizer, batch_size)
        test_loss = test(model, test_dataset, batch_size)
        print(
            f"Epoch [{epoch + 1}/{epochs}] -- loss: {train_loss} - test_loss: {test_loss}"
        )
    return model


def run(
    data_path: str,
    model_path: str,
    kernel_size: int = 5,
    epochs: int = 100,
    batch_size: int = 64,
    train_test_ratio: float = 0.8,
):
    print(f"data_path: {data_path}")
    print(f"model_path: {model_path}")
    print(f"kernel_size: {kernel_size}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"train_test_ratio: {train_test_ratio}")
    print("-" * 40)

    dataset = WeatherDataset(data_path)
    (train_dataset, test_dataset) = split_dataset(dataset, train_test_ratio)
    print(f"Train dataset contains {len(train_dataset)} examples")
    print(f"Test dataset contains {len(test_dataset)} examples")

    (std, mean) = std_mean(train_dataset)
    print(f"Train dataset std: {std.shape}")
    print(std.reshape(std.shape[0]))
    print(f"Train dataset mean: {mean.shape}")
    print(mean.reshape(mean.shape[0]))

    print(f"Device: {DEVICE}")

    model = Model(std, mean, kernel_size).to(DEVICE)
    print(model)

    trained_model = fit(model, train_dataset, test_dataset, epochs, batch_size)
    torch.save(trained_model, model_path)
    print(f"Model saved to path: {model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-test-ratio", type=float, default=0.8)
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        model_path=args.model_path,
        kernel_size=args.kernel_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_test_ratio=args.train_test_ratio,
    )
