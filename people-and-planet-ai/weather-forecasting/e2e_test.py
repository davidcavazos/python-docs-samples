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

from typing import Iterable

import nbformat
from nbclient import NotebookClient


def patch_code_cell(source: str) -> Iterable[str]:
    if "from google.colab import auth" in source:
        yield "import sys"
        yield "from unittest.mock import Mock"
        yield "sys.modules['google.colab'] = Mock()"
    yield source


def on_cell_execute(cell: nbformat.NotebookNode, cell_index: int):
    cell["source"] = "\n".join(patch_code_cell(cell["source"]))


def test_notebook():
    nb = nbformat.read("README.ipynb", as_version=4)
    client = NotebookClient(nb, timeout=600)
    client.on_cell_execute = on_cell_execute
    client.execute()


if __name__ == "__main__":
    test_notebook()
