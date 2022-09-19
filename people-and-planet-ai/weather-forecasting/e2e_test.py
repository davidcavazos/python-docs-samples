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

import re
from typing import Iterable

import nbformat
from nbclient import NotebookClient


prelude = [
    "from unittest.mock import Mock",
    "import sys",
    "sys.modules['google.colab'] = Mock()",
    "exit = Mock()",
]


# https://regex101.com/r/auHETK/1
FILTER_SHELL_COMMANDS = re.compile(r"^!(?:[^\n]+\\\n)*(?:[^\n]+)$", re.MULTILINE)


def on_cell_execute(cell_index: int, cell: nbformat.NotebookNode):
    cell["source"] = FILTER_SHELL_COMMANDS.sub("", cell["source"])


def test_notebook():
    nb = nbformat.read("README.ipynb", as_version=4)
    nb.cells = [nbformat.v4.new_code_cell("\n".join(prelude))] + nb.cells
    client = NotebookClient(nb, timeout=600)
    client.on_cell_execute = on_cell_execute
    client.execute()


if __name__ == "__main__":
    test_notebook()
