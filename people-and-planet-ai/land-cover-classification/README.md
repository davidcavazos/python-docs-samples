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
pip install src/inputs src/dataset
python -m landcover.dataset.create data/np --direct_num_workers=20 --direct_running_mode=multi_threading

pip install src/inputs src/dataset src/tensorflow/model
python -m landcover.dataset.create data/tf --direct_num_workers=20 --direct_running_mode=multi_threading --tfrecords

# Create dataset (Dataflow)
export PROJECT="My Google Cloud Project ID"
export BUCKET="My Cloud Storage bucket name"
export LOCATION="us-central1"

pip install src/inputs src/dataset
python -m build --sdist src/inputs/ --outdir build/
python -m build --sdist src/dataset/ --outdir build/

python -m landcover.dataset.create gs://$BUCKET/landcover/data/np \
  --runner="DataflowRunner" \
  --project="$PROJECT" \
  --region="$LOCATION" \
  --temp_location="gs://$BUCKET/temp" \
  --requirements_cache="skip" \
  --extra_package="build/landcover-inputs-1.0.0.tar.gz" \
  --extra_package="build/landcover-dataset-1.0.0.tar.gz"

pip install src/inputs src/dataset
python -m build --sdist src/inputs/ --outdir build/
python -m build --sdist src/dataset/ --outdir build/
python -m build --sdist src/tensorflow/model/ --outdir build/

python -m landcover.dataset.create gs://$BUCKET/landcover/data/tf \
  --tfrecords \
  --runner="DataflowRunner" \
  --project="$PROJECT" \
  --region="$LOCATION" \
  --temp_location="gs://$BUCKET/temp" \
  --requirements_cache="skip" \
  --extra_package="build/landcover-inputs-1.0.0.tar.gz" \
  --extra_package="build/landcover-dataset-1.0.0.tar.gz" \
  --extra_package="build/landcover-tensorflow-model-1.0.0.tar.gz"

# Train model (local)
pip install "src/landcover" "src/tensorflow/trainer"

# Train model (Vertex AI)
python -m build --sdist src/landcover/ --outdir build/
python -m build --sdist src/tensorflow/trainer/ --outdir build/

gsutil cp build/landcover*.tar.gz gs://$BUCKET/landcover/
gsutil cp build/trainer*.tar.gz gs://$BUCKET/landcover/

# Get predictions
```
