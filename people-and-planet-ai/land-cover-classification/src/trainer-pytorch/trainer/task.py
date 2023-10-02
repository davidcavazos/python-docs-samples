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

import os

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from trainer.dataset import LandCoverDataset
from trainer.model import LandCoverModel

from landcover.inputs import LANDCOVER_CLASSES

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("model_dir")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-split", type=float, default=0.9)
    args = parser.parse_args()

    assert (
        0 < args.train_split < 1
    ), f"--train-split must be between 0 and 1, got {args.train_split}"
    splits = [args.train_split, 1 - args.train_split]

    dataset = LandCoverDataset(args.data_dir)
    print(f"{len(dataset)=}")
    train_subset, val_subset = random_split(dataset, splits)
    print(f"{len(train_subset)=}")
    print(f"{len(val_subset)=}")

    train_inputs = np.stack([x for x, _ in train_subset])
    mean = train_inputs.mean(axis=(0, 2, 3)).tolist()
    print(f"{len(mean)=}")
    print(f"{mean=}")
    std = train_inputs.std(axis=(0, 2, 3)).tolist()
    print(f"{len(std)=}")
    print(f"{std=}")

    model = LandCoverModel(mean, std, len(LANDCOVER_CLASSES))
    trainer = pl.Trainer(
        default_root_dir="model/pytorch",
        max_epochs=args.max_epochs,
        logger=True,
    )

    train_dataloader = DataLoader(
        train_subset,
        args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count() or 0,
        persistent_workers=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_subset,
        args.batch_size,
        num_workers=os.cpu_count() or 0,
        persistent_workers=True,
        pin_memory=True,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    model.save(args.model_dir)
