python create-datasets.py \
  --output-path "gs://$BUCKET/weather-forecasting/data" \
  --num-dates 1 \
  --points-per-date 1 \
  --runner "DataflowRunner" \
  --project "$PROJECT" \
  --region "$REGION" \
  --temp_location "gs://$BUCKET/weather-forecasting/temp"