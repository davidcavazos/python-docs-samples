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

from glob import glob
import os

import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

from landcover.inputs import LANDCOVER_NAME


class LandCoverDataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        self.files = glob(os.path.join(data_dir, "*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        self.chunk_sizes = [len(np.load(f)) for f in self.files]
        self.length = sum(self.chunk_sizes)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seen = 0
        for size, filename in zip(self.chunk_sizes, self.files):
            if idx - seen < size:
                example = np.load(filename)[(str(idx - seen))]
                inputs = [
                    example[field]
                    for field in example.dtype.names
                    if field != LANDCOVER_NAME
                ]
                labels = [example[LANDCOVER_NAME]]
                return torch.tensor(np.array(inputs)), torch.tensor(np.array(labels))
            seen += size
        return (torch.tensor([]), torch.tensor([]))
