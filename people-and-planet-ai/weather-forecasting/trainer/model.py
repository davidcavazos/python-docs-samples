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

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset

# Default values.
EPOCHS = 2
BATCH_SIZE = 256
TRAIN_TEST_RATIO = 0.9

# Constants.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WeatherDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        self.files = glob(os.path.join(data_path, "*.npz"))

        if not self.files:
            raise FileNotFoundError(f"No files found in dataset: {data_path}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        with open(self.files[idx], "rb") as f:
            npz = np.load(f)
            # Convert to channels-last format since that's what PyTorch expects.
            inputs = torch.from_numpy(npz["inputs"]).transpose(0, -1)
            labels = torch.from_numpy(npz["labels"]).transpose(0, -1)
            return (inputs, labels)


# https://developers.google.com/machine-learning/data-prep/transform/normalization#z-score
class Normalization(torch.nn.Module):
    """Preprocessing normalization layer with z-score."""

    def __init__(self, std: torch.Tensor, mean: torch.Tensor) -> None:
        super().__init__()
        self.std = std.to(DEVICE, non_blocking=True)
        self.mean = mean.to(DEVICE, non_blocking=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    @staticmethod
    def adapt(dataset: Dataset, batch_size: int = 32, axis: int = -1) -> Normalization:
        (first, _) = dataset[0]
        idx = axis if axis >= 0 else first.dim() + axis
        dims = [i for i in range(first.dim() + 1) if i != idx + 1]
        shape = [x if i == idx else 1 for i, x in enumerate(first.shape)]

        size = 0
        sum = torch.zeros(shape).to(DEVICE, non_blocking=True)
        sum_sq = torch.zeros(shape).to(DEVICE, non_blocking=True)
        for inputs, _ in data_loader(dataset, batch_size):
            inputs = inputs.to(DEVICE, non_blocking=True)
            sum += inputs.sum(dims).reshape(shape)
            sum_sq += (inputs**2).sum(dims).reshape(shape)
            size += len(inputs)
        mean = sum / size
        variance = sum_sq / size - mean**2
        std = torch.sqrt(torch.abs(variance))
        return Normalization(std, mean)


class Model(torch.nn.Module):
    def __init__(self, normalization: Normalization) -> None:
        super().__init__()
        inputs = 52
        hidden1 = 16
        hidden2 = 4
        outputs = 2
        kernel_size = (5, 5)

        self.normalization = normalization
        self.layers = torch.nn.Sequential(
            self.normalization,
            torch.nn.Conv2d(inputs, hidden1, kernel_size),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(hidden1, hidden2, kernel_size),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden2, outputs, (1, 1)),
            torch.nn.ReLU(),  # precipitation cannot be negative
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ys = self.layers(x)
        y1 = ys[0:1]
        y2 = ys[1:2]
        return y1, y2

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.predict_batch(np.stack([inputs]))[0]

    def predict_batch(self, inputs: np.ndarray) -> np.ndarray:
        channels_first_inputs = np.moveaxis(inputs, -1, 1)
        inputs_pt = torch.from_numpy(channels_first_inputs).to(DEVICE)
        with torch.no_grad():
            return np.moveaxis(self(inputs_pt).cpu().numpy(), 1, -1)

    def save(self, model_path: str) -> None:
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.normalization.std, os.path.join(model_path, "std.pt"))
        torch.save(self.normalization.mean, os.path.join(model_path, "mean.pt"))
        torch.save(self.state_dict(), os.path.join(model_path, "state_dict.pt"))

    @staticmethod
    def load(model_path: str) -> Model:
        std = torch.load(os.path.join(model_path, "std.pt"), map_location=DEVICE)
        mean = torch.load(os.path.join(model_path, "mean.pt"), map_location=DEVICE)
        state_dict = torch.load(
            os.path.join(model_path, "state_dict.pt"), map_location=DEVICE
        )
        model = Model(Normalization(std, mean))
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(DEVICE)


def train_test_split(dataset: WeatherDataset, ratio: float) -> list[Subset]:
    train_size = int(len(dataset) * ratio)
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def data_loader(dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size,
        num_workers=os.cpu_count() or 0,
        shuffle=shuffle,
        pin_memory=True,
        persistent_workers=True,
    )


def train(model: Model, loader: DataLoader, loss: torch.nn.Module) -> float:
    optimizer = torch.optim.Adam(model.parameters())

    total_loss = 0.0
    model.train()
    for inputs_batch, labels_batch in loader:
        inputs_batch = inputs_batch.to(DEVICE, non_blocking=True)
        labels_batch = labels_batch.to(DEVICE, non_blocking=True)

        # Compute prediction error.
        predictions1, predictions2 = model(inputs_batch)
        loss1 = loss(predictions1, labels_batch[0:1])
        loss2 = loss(predictions2, labels_batch[1:2])
        batch_loss = loss1 + loss2
        total_loss += batch_loss.item()

        # Backpropagation.
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    return total_loss / len(loader)


def test(model: Model, loader: DataLoader, loss: torch.nn.Module) -> float:
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs_batch, labels_batch in loader:
            inputs_batch = inputs_batch.to(DEVICE, non_blocking=True)
            labels_batch = labels_batch.to(DEVICE, non_blocking=True)

            predictions1, predictions2 = model(inputs_batch)
            loss1 = loss(predictions1, labels_batch[0:1])
            loss2 = loss(predictions2, labels_batch[1:2])
            total_loss += (loss1 + loss2).item()
    return total_loss / len(loader)


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
    print(f"DEVICE: {DEVICE}")
    print("-" * 40)

    dataset = WeatherDataset(data_path)
    (train_data, test_data) = train_test_split(dataset, train_test_ratio)
    print(f"Train dataset contains {len(train_data)} examples")
    print(f"Test dataset contains {len(test_data)} examples")

    normalization = Normalization.adapt(train_data, batch_size, axis=0)
    print(f"Train dataset mean: {normalization.mean.shape}")
    print(f"Train dataset std:  {normalization.std.shape}")

    train_loader = data_loader(train_data, batch_size, shuffle=True)
    test_loader = data_loader(test_data, batch_size)

    model = Model(normalization).to(DEVICE)
    print(model)

    loss = torch.nn.SmoothL1Loss()
    print(f"loss: {loss}")
    for epoch in range(epochs):
        train_loss = train(model, train_loader, loss)
        test_loss = test(model, test_loader, loss)
        print(
            f"Epoch [{epoch + 1}/{epochs}] -- "
            f"train_loss: {train_loss:.4f} - "
            f"test_loss: {test_loss:.4f}"
        )

    model.save(model_path)
    print(f"Model saved to path: {model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--train-test-ratio", type=float, default=TRAIN_TEST_RATIO)
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_test_ratio=args.train_test_ratio,
    )
