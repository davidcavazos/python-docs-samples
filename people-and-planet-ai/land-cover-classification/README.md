# 🌍 Land cover classification -- _image segmentation_

> ⚠️ This sample is currently not fully working. We're working on refactoring it to make it simpler and fix the issues.

## [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GoogleCloudPlatform/python-docs-samples/blob/main/people-and-planet-ai/land-cover-classification/README.ipynb) 🌍 TensorFlow with Earth Engine introduction

## [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GoogleCloudPlatform/python-docs-samples/blob/main/people-and-planet-ai/land-cover-classification/cloud-tensorflow.ipynb) ☁️ Scaling TensorFlow with Cloud

## [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GoogleCloudPlatform/python-docs-samples/blob/main/people-and-planet-ai/land-cover-classification/land-cover-change.ipynb) 🗺️ Visualizing land cover change

> [Watch the video in YouTube<br> ![thumbnail](http://img.youtube.com/vi/zImQf91ffFo/0.jpg)](https://youtu.be/zImQf91ffFo)

This model uses satellite data to classify what is on Earth. The satellite data comes from [Google Earth Engine.](https://earthengine.google.com/)

* **Model**: 2D Fully Convolutional Network in [TensorFlow]
* **Creating datasets**: [Sentinel-2] satellite data and [ESA WorldCover] from [Earth Engine] with [Dataflow]
* **Training the model**: [TensorFlow] in [Vertex AI]
* **Getting predictions**: [TensorFlow] in [Cloud Run] (real-time) and [Dataflow] (batch)

[Sentinel-2]: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2
[ESA WorldCover]: https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100

[Cloud Run]: https://cloud.google.com/run
[Dataflow]: https://cloud.google.com/dataflow
[Earth Engine]: https://earthengine.google.com/
[TensorFlow]: https://www.tensorflow.org/
[Vertex AI]: https://cloud.google.com/vertex-ai

```sh
# Create dataset (local)
pip install "src/landcover[build,dataset]" "src/trainer"
python -m landcover.dataset.create data/

# Create dataset (Dataflow)
export PROJECT="My Google Cloud Project ID"
export BUCKET="My Cloud Storage bucket name"
export LOCATION="us-central1"

python -m build --sdist src/landcover/ --outdir build/
python -m landcover.dataset.create gs://$BUCKET/landcover/data \
  --runner="DataflowRunner" \
  --project="$PROJECT" \
  --region="$LOCATION" \
  --temp_location="gs://$BUCKET/temp" \
  --requirements_cache="skip" \
  --extra_package="build/landcover-1.0.0.tar.gz"

# Train model (local)
pip install "src/landcover" "src/trainer-tensorflow"

# Train model (Vertex AI)
python -m build --sdist src/landcover/ --outdir build/
python -m build --sdist src/trainer-tensorflow/ --outdir build/



# Get predictions
```
