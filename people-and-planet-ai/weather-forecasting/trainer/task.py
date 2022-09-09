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
from typing import Tuple
import os

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, random_split


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
        self.std = std
        self.mean = mean

    def forward(self, x):
        return (x - self.mean) / self.std


class FullyConvolutionalNetwork(torch.nn.Module):
    def __init__(
        self,
        std: torch.Tensor,
        mean: torch.Tensor,
        num_inputs: int,
        num_outputs: int,
        patch_size: int,
        input_timesteps: int,
        output_timesteps: int,
        hidden_units: int = 8,
    ):
        super(FullyConvolutionalNetwork, self).__init__()
        self.layers = torch.nn.Sequential(
            Normalize(std, mean),
            torch.nn.Conv3d(
                in_channels=num_inputs,
                out_channels=hidden_units,
                kernel_size=(input_timesteps, patch_size, patch_size),
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(
                in_channels=hidden_units,
                out_channels=num_outputs,
                kernel_size=(output_timesteps, patch_size, patch_size),
            ),
        )
        self.loss = torch.nn.SmoothL1Loss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


def std_mean(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    num_channels = dataset[0][0].shape[0]
    shape = [num_channels] + [1] * len(dataset[0][0].shape[1:])
    total_sum = torch.zeros(shape)
    squared_sum = torch.zeros(shape)
    for inputs, _ in dataset:
        total_sum += torch.mean(inputs)
        squared_sum += torch.mean(inputs**2)
    mean = total_sum / len(dataset)
    std = (squared_sum / len(dataset) - mean**2) ** 0.5
    return (std, mean)


def split_dataset(dataset: Dataset, train_test_ratio: float) -> Tuple[Dataset, Dataset]:
    train_size = int(len(dataset) * train_test_ratio)
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def train(
    model: FullyConvolutionalNetwork,
    dataset: Dataset,
    optimizer: Optimizer,
    batch_size: int,
) -> float:
    dataloader = DataLoader(
        dataset, batch_size, num_workers=os.cpu_count(), shuffle=True
    )
    total_loss = 0.0

    model.train()
    for inputs, labels in dataloader:
        (inputs, labels) = (inputs.to(model.device), labels.to(model.device))

        # Compute prediction error
        predictions = model(inputs)
        loss = model.loss(predictions, labels)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)


def test(model: FullyConvolutionalNetwork, dataset: Dataset, batch_size: int) -> float:
    dataloader = DataLoader(dataset, batch_size, num_workers=os.cpu_count())
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            (inputs, labels) = (inputs.to(model.device), labels.to(model.device))

            predictions = model(inputs)
            total_loss += model.loss(predictions, labels).item()
    return total_loss / len(dataloader)


def fit(
    model: FullyConvolutionalNetwork,
    train_dataset: Dataset,
    test_dataset: Dataset,
    optimizer: Optimizer,
    batch_size: int,
    epochs: int,
) -> FullyConvolutionalNetwork:
    print(f"Device: {model.device}")
    for epoch in range(epochs):
        train_loss = train(model, train_dataset, optimizer, batch_size)
        test_loss = test(model, test_dataset, batch_size)
        print(
            f"Epoch [{epoch + 1}/{epochs}] -- loss: {train_loss:.2f} - test_loss: {test_loss:.2f}"
        )
    return model


def run(
    data_path: str,
    model_path: str,
    batch_size: int,
    epochs: int,
    train_test_ratio: float,
    patch_size: int,
    input_timesteps: int,
    output_timesteps: int,
):
    dataset = WeatherDataset(data_path)
    (train_data, test_data) = split_dataset(dataset, train_test_ratio)
    print(f"Train dataset contains {len(train_data)} examples")
    print(f"Test dataset contains {len(test_data)} examples")

    (std, mean) = std_mean(train_data)
    print(f"std: {std.numpy()}")
    print(f"mean: {mean.numpy()}")

    model = FullyConvolutionalNetwork(
        std=std,
        mean=mean,
        num_inputs=17,
        num_outputs=1,
        patch_size=patch_size,
        input_timesteps=input_timesteps,
        output_timesteps=output_timesteps,
    )
    print(f"Model device: {model.device}")
    print(model)

    optimizer = torch.optim.Adam(model.parameters())
    print(optimizer)
    trained_model = fit(model, train_data, test_data, optimizer, batch_size, epochs)

    torch.save(trained_model, model_path)
    print(f"Model saved to path: {model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-test-ratio", type=float, default=0.9)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--input-timesteps", type=int, default=3)
    parser.add_argument("--output-timesteps", type=int, default=2)
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_test_ratio=args.train_test_ratio,
        patch_size=args.patch_size,
        input_timesteps=args.input_timesteps,
        output_timesteps=args.output_timesteps,
    )
