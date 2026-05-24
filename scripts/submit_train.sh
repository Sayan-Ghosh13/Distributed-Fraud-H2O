#!/usr/bin/env bash
set -e

MASTER_URL=${MASTER_URL:-spark://bhaskara16:7077}
INPUT_PATH=${INPUT_PATH:-hdfs:///project/data/creditcard.csv}
MODEL_DIR=${MODEL_DIR:-hdfs:///project/models/h2o_rf}
PRED_OUT=${PRED_OUT:-hdfs:///project/output/test_predictions}
APP_NAME=${APP_NAME:-FraudDetectionSparkH2O}

spark-submit \
  --master "$MASTER_URL" \
  --deploy-mode client \
  --name "$APP_NAME" \
  src/train_fraud_h2o.py \
    --input "$INPUT_PATH" \
    --model-dir "$MODEL_DIR" \
    --prediction-output "$PRED_OUT" \
    --repartition 8 \
    --ntrees 200 \
    --max-depth 20 \
    --sample-rate 0.80 \
    --seed 42