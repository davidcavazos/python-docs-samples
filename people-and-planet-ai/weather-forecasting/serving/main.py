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

import flask

from . import data

app = flask.Flask(__name__)

try:
    data.ee_init()
    ee_status = "✅ I was able to authenticate to Earth Engine!"
except Exception as e:
    ee_status = f"❌ I couldn't authenticate to Earth Engine 😔\n{e}"


@app.route("/", methods=["POST"])
def ping() -> dict:
    try:
        args = flask.request.get_json() or {}
        args_status = "✅ I was able to parse your JSON arguments!"
    except Exception as e:
        args = None
        args_status = f"❌ I couldn't find any JSON arguments 😔\n{e}"

    return {
        "args_status": args_status,
        "args_received": args,
        "ee_status": ee_status,
    }


if __name__ == "__main__":
    import os

    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
