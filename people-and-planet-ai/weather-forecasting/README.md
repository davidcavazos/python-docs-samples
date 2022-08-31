# Google Cloud resources

```sh
PROJECT="my-cloud-project-name"
BUCKET="my-cloud-storage-bucket"
LOCATION="my-cloud-location"
```

# Create the datasets

```sh
python datasets.py \
  --output-path "gs://$BUCKET/weather-forecasting/data" \
  --num-dates "24" \
  --points-per-date "20" \
  --runner "DataflowRunner" \
  --project "$PROJECT" \
  --region "$LOCATION" \
  --temp_location "gs://$BUCKET/weather-forecasting/temp"
```

https://console.cloud.google.com/dataflow/jobs

# Train the model

```sh
# https://cloud.google.com/vertex-ai/docs/training/pre-built-containers#pytorch
gcloud ai custom-jobs local-run \
  --executor-image-uri "us-docker.pkg.dev/vertex-ai/training/pytorch-xla.1-11:latest" \
  --script "trainer.py"

# https://cloud.google.com/vertex-ai/docs/training/create-custom-job#create_custom_job-gcloud
# https://cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/create
gcloud ai custom-jobs create \
  --display-name "weather-forecasting" \
  --config "trainer-config.yaml" \
  --region "$LOCATION"
```

https://console.cloud.google.com/vertex-ai/training/custom-jobs
