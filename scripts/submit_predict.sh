#!/usr/bin/env bash
set -e

MASTER_URL=${MASTER_URL:-spark://bhaskara16:7077}
INPUT_PATH=${INPUT_PATH:-hdfs:///project/data/creditcard_unseen.csv}
MODEL_PATH=${MODEL_PATH:-/tmp/h2o_model_path_from_training}
OUTPUT_PATH=${OUTPUT_PATH:-hdfs:///project/output/new_predictions}
APP_NAME=${APP_NAME:-FraudPredictionSparkH2O}

spark-submit \
  --master "$MASTER_URL" \
  --deploy-mode client \
  --name "$APP_NAME" \
  src/predict_fraud_h2o.py \
    --input "$INPUT_PATH" \
    --model-path "$MODEL_PATH" \
    --output "$OUTPUT_PATH" \
    --repartition 8