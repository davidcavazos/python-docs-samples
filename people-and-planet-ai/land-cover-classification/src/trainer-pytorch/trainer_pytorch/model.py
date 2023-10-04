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

from __future__ import annotations

import json
import os

import lightning.pytorch as pl
import torch
import torch.nn as T
import torch.nn.functional as F
import torchmetrics.functional as metrics

NUM_CLASSES = 9


class LandCoverModel(pl.LightningModule):
    def __init__(
        self,
        mean: list[float],
        std: list[float],
        kernel_size: int = 5,
        hidden1: int = 32,
        hidden2: int = 64,
    ):
        super().__init__()
        self.config = {
            "mean": mean,
            "std": std,
            "kernel_size": kernel_size,
            "hidden1": hidden1,
            "hidden2": hidden2,
        }

        self.layers = torch.nn.Sequential(
            Normalize(mean, std),
            T.Conv2d(len(mean), hidden1, kernel_size=kernel_size),
            T.ReLU(),
            T.ConvTranspose2d(hidden1, hidden2, kernel_size=kernel_size),
            T.ReLU(),
            T.Conv2d(hidden2, NUM_CLASSES, kernel_size=1),
            T.Softmax(dim=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs, torch.argmax(targets, dim=1))
        self.log_dict(
            {
                "loss": loss,
                "accuracy": metrics.accuracy(
                    torch.argmax(outputs, dim=1),
                    torch.argmax(targets, dim=1),
                    task="multiclass",
                    num_classes=NUM_CLASSES,
                ),
            }
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.layers.parameters(), lr=0.001)

    def save(self, model_dir: str) -> None:
        model_path = model_dir.replace("gs://", "/gcs/")
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        torch.save(self.state_dict(), os.path.join(model_path, "state_dict.pt"))

    @staticmethod
    def load(model_dir: str) -> LandCoverModel:
        model_path = model_dir.replace("gs://", "/gcs/")
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)
        state_dict = torch.load(os.path.join(model_path, "state_dict.pt"))

        model = LandCoverModel(**config)
        model.load_state_dict(state_dict)
        model.eval()
        return model


class Normalize(torch.nn.Module):
    def __init__(self, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.mean = torch.nn.Parameter(torch.tensor(mean).view(-1, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor(std).view(-1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
