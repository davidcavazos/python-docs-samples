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
# On MacOS with M1 chips.
export GRPC_PYTHON_LDFLAGS=" -framework CoreFoundation"
pip install --no-cache-dir --force-reinstall --no-binary :all: grpcio
```

```sh
# Create dataset (local)
pip install src/inputs src/dataset
python -m landcover.dataset.create data/np --direct_num_workers=20 --direct_running_mode=multi_threading

pip install src/inputs src/dataset src/model/tensorflow
python -m landcover.dataset.create data/tf --direct_num_workers=20 --direct_running_mode=multi_threading --tfrecords

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

#  100 samples: 1 worker, 18m -- https://pantheon.corp.google.com/dataflow/jobs/us-central1/2023-09-13_14_12_36-5090516709424439239
#   1K samples: 3 workers, 18m -- https://pantheon.corp.google.com/dataflow/jobs/us-central1/2023-09-13_14_33_17-10394742472595658906
#  10K samples: 2 workers, 47m -- https://pantheon.corp.google.com/dataflow/jobs/us-central1/2023-09-13_14_52_00-16553727520001131852
# 100K samples: 21->10 workers, 1h 38m -- https://pantheon.corp.google.com/dataflow/jobs/us-central1/2023-09-13_15_47_18-9227384235400719392
#   1M samples: 20->10 workers, 8h 25m -- https://pantheon.corp.google.com/dataflow/jobs/us-central1/2023-09-13_17_54_52-10006425792414329007
#   1M (unthrottled): 20->10 workers, 10h 4m -- https://pantheon.corp.google.com/dataflow/jobs/us-central1/2023-09-14_11_30_56-16263900987840015973
#  10M samples: workers, h m -- https://pantheon.corp.google.com/dataflow/jobs/us-central1/2023-09-18_17_44_29-16480442746729276935

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

# -----------------

# Train model (local)
pip install build src/model/tensorflow src/trainer/tensorflow
python -m trainer.task data/tf-1M model --tensorboard="logs/tensorflow"

# Train model (Vertex AI)
python -m build --sdist src/model/tensorflow --outdir build/
python -m build --sdist src/trainer/tensorflow --outdir build/

gsutil cp build/model-tensorflow-*.tar.gz gs://$BUCKET/landcover/
gsutil cp build/trainer-tensorflow-*.tar.gz gs://$BUCKET/landcover/

# -----------------

# Get predictions (local)
python src/serving/tensorflow/main.py

# Get predictions (emulator)
$(cd src/serving/tensorflow; gcloud beta code dev)

# Get predictions (Cloud Run)
gcloud run deploy landcover-tensorflow src/serving/tensorflow
```
