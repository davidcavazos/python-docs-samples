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

"""Web service to host model predictions."""

import json
import logging
import os

import flask
import numpy as np
import tensorflow as tf

from landcover.inputs import ee_init

app = flask.Flask(__name__)

# Set this environment variable when deploying the model.
MODEL: tf.keras.Model = tf.keras.models.load_model(os.environ["MODEL_PATH"])

ee_init()


@app.route("/")
def ping() -> dict:
    """Check that we can communicate with the service and get arguments."""
    return {
        "response": "✅ I got your request!",
        "args": flask.request.args,
    }


@app.route("/predict/<int:year>/@<float(signed=True):lon>,<float(signed=True):lat>")
def predict(lon: float, lat: float, year: int) -> flask.Response:
    """Gets a prediction from the model.

    Args:
        year: Year of interest, a median composite is used.
        lon: Longitude of the point of interest.
        lat: Latitude of the point of interest.

    Optional query parameters:
        patch-size: Size in pixels of the surrounding square patch.

    Returns:
        A JSON response with the predictions if successful, or an error otherwise.
    """

    # Optional HTTP request parameters.
    #   https://en.wikipedia.org/wiki/Query_string
    patch_size = flask.request.args.get("patch-size", 512, type=int)

    try:
        # Get predictions from the model.
        inputs = data.get_input_patch(year, (lon, lat), patch_size)
        inputs_batch = np.stack([inputs])
        probabilities = MODEL.predict(inputs_batch)[0]
        predictions = probabilities.argmax(axis=-1).astype(np.uint8)

        # Return the model predictions.
        return flask.make_response({"predictions": predictions.tolist()})

    # Anything could go wrong in Python, so we protect the server against
    # any exception and send a valid response with a human-readable error
    # message, instead of a generic "500: Internal Server Error".
    except Exception as e:
        # Log the error, and return a valid JSON response with status 500.
        logging.exception(e)
        return ({"error": f"{type(e).__name__}: {e}"}, 500)


if __name__ == "__main__":
    # Run for local debugging, this is not meant for production.
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
