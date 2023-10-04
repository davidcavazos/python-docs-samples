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

## Install

```sh
# On MacOS with M1 chips.
export GRPC_PYTHON_LDFLAGS=" -framework CoreFoundation"
pip install --no-cache-dir --force-reinstall --no-binary :all: grpcio
```

## Create dataset

```sh
# TensorFlow local
pip install src/inputs src/dataset src/model/tensorflow
python -m landcover.dataset.create data/tf --direct_num_workers=20 --direct_running_mode=multi_threading --tfrecords

# PyTorch local
pip install src/inputs src/dataset
python -m landcover.dataset.create data/np --direct_num_workers=20 --direct_running_mode=multi_threading

# Create dataset (Dataflow)
export PROJECT="My Google Cloud Project ID"
export BUCKET="My Cloud Storage bucket name"
export LOCATION="us-central1"

pip install build src/inputs src/dataset
python -m build --sdist src/inputs/ --outdir build/
python -m build --sdist src/dataset/ --outdir build/

python -m landcover.dataset.create gs://$BUCKET/landcover/data/np-1M-unthrottled \
  --num-samples=1000000 \
  --runner="DataflowRunner" \
  --project="$PROJECT" \
  --region="$LOCATION" \
  --temp_location="gs://$BUCKET/temp" \
  --requirements_cache="skip" \
  --extra_package="build/landcover-inputs-1.0.0.tar.gz" \
  --extra_package="build/landcover-dataset-1.0.0.tar.gz"

pip install src/inputs src/dataset src/model/tensorflow
python -m build --sdist src/inputs --outdir build/
python -m build --sdist src/dataset --outdir build/
python -m build --sdist src/model/tensorflow --outdir build/

python -m landcover.dataset.create gs://$BUCKET/landcover/data/tf \
  --tfrecords \
  --runner="DataflowRunner" \
  --project="$PROJECT" \
  --region="$LOCATION" \
  --temp_location="gs://$BUCKET/temp" \
  --requirements_cache="skip" \
  --extra_package="build/landcover-inputs-1.0.0.tar.gz" \
  --extra_package="build/landcover-dataset-1.0.0.tar.gz" \
  --extra_package="build/landcover-model-tensorflow-1.0.0.tar.gz"

python -m landcover.dataset.create /tmp/data/tmp

# Convert np to tf
pip install src/dataset src/model/tensorflow
python -m build --sdist src/dataset/ --outdir build/
python -m build --sdist src/model/tensorflow --outdir build/

python -m landcover.dataset.np_to_tf gs://$BUCKET/landcover/data/np-1M gs://$BUCKET/landcover/data/tf-1M --direct_num_workers=20 --direct_running_mode=multi_threading

```

# Train the model

```sh
# PyTorch local
pip install src/trainer-pytorch
python -m trainer_pytorch.task data/np-10K model/pytorch

# PyTorch Vertex AI
python -m build --sdist src/trainer-pytorch --outdir build/
gsutil cp build/trainer-pytorch-1.0.0.tar.gz gs://$BUCKET/landcover/
VERSION="10K"
python -m trainer_pytorch.vertex gs://$BUCKET/landcover/data/np-$VERSION gs://$BUCKET/landcover/model/pytorch-$VERSION \
  --display-name="PT land-cover-$VERSION" \
  --project="$PROJECT" \
  --bucket="$BUCKET" \
  --package="gs://$BUCKET/landcover/trainer-pytorch-1.0.0.tar.gz"

# TensorFlow local
pip install src/trainer-tensorflow
python -m trainer_tensorflow.task data/tf-10K model/tensorflow

# TensorFlow Vertex AI
python -m build --sdist src/trainer-tensorflow --outdir build/
gsutil cp build/trainer-tensorflow-1.0.0.tar.gz gs://$BUCKET/landcover/
VERSION="1M"
python -m trainer_tensorflow.vertex gs://$BUCKET/landcover/data/tf-$VERSION gs://$BUCKET/landcover/model/tensorflow-$VERSION \
  --display-name="TF land-cover-$VERSION" \
  --project="$PROJECT" \
  --bucket="$BUCKET" \
  --package="gs://$BUCKET/landcover/trainer-tensorflow-1.0.0.tar.gz"
```

## Predictions

```sh
MODEL="gs://$BUCKET/landcover/model/tensorflow-10K"
```
