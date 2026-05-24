import argparse
import socket
import time

import h2o
from pyspark import TaskContext
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.storagelevel import StorageLevel
from pysparkling import *
from pyspark import TaskContext


LABEL_COL = "Class"
ROW_ID_COL = "__row_id"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict credit card fraud using saved H2O model on Spark cluster"
    )
    parser.add_argument("--input", required=True, help="Input CSV path for unseen data")
    parser.add_argument("--model-path", required=True, help="Saved H2O model path")
    parser.add_argument("--output", required=True, help="Output path for predictions")
    parser.add_argument("--repartition", type=int, default=8, help="Number of Spark partitions")
    return parser.parse_args()


def build_spark(app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.ext.h2o.client.language", "python")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def worker_partition_logger(iter_rows):
    host = socket.gethostname()
    tc = TaskContext.get()
    partition_id = tc.partitionId() if tc else -1
    start = time.time()

    count = 0
    for row in iter_rows:
        count += 1

    end = time.time()
    msg = (
        f"[WORKER LOG] Host={host} | Partition={partition_id} | "
        f"Rows={count} | Status=DONE | Time={end - start:.2f}s"
    )
    log_file = f"/tmp/spark_worker_{host}.log"
    with open(log_file, "a") as f:
        f.write(msg)
    print(msg)
    yield (host, partition_id, count, end - start, "DONE")


def print_partition_completion(df, stage_name: str):
    print(f"\n===== START WORKER STAGE: {stage_name} =====")
    results = df.rdd.mapPartitions(worker_partition_logger).collect()
    print(f"===== COMPLETED WORKER STAGE: {stage_name} =====")
    for host, partition_id, count, seconds, status in results:
        print(
            f"[DRIVER SUMMARY] Stage={stage_name} | Worker={host} | "
            f"Partition={partition_id} | Rows={count} | Status={status} | Time={seconds:.2f}s"
        )


def cast_numeric_columns(df: DataFrame) -> DataFrame:
    casted_cols = []
    for c in df.columns:
        if c == LABEL_COL:
            casted_cols.append(F.col(c).cast(T.IntegerType()).alias(c))
        else:
            casted_cols.append(F.col(c).cast(T.DoubleType()).alias(c))
    return df.select(*casted_cols)


def clean_dataframe(df: DataFrame) -> DataFrame:
    fill_dict = {}
    for c in df.columns:
        if c != LABEL_COL:
            fill_dict[c] = 0.0
    return df.fillna(fill_dict)


def add_row_id(df: DataFrame) -> DataFrame:
    return df.withColumn(ROW_ID_COL, F.monotonically_increasing_id())


def main():
    args = parse_args()

    spark = build_spark("Predict-CreditCard-Fraud-H2O")
    hc = H2OContext.getOrCreate()

    print(f"Reading data from: {args.input}")
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(args.input)
    )

    df = cast_numeric_columns(df)
    df = clean_dataframe(df)
    df = add_row_id(df)

    if args.repartition > 0:
        df = df.repartition(args.repartition)

    df.persist(StorageLevel.MEMORY_AND_DISK)
    df.rdd.mapPartitions(worker_partition_logger).collect()

    print_partition_completion(df, "PREDICT-INPUT-READY")

    print(f"Loading model from: {args.model_path}")
    model = h2o.load_model(args.model_path)

    h2o_frame = hc.asH2OFrame(df, h2oFrameName="prediction_input")

    feature_cols = [c for c in df.columns if c not in [LABEL_COL, ROW_ID_COL]]

    if LABEL_COL in df.columns:
        h2o_frame[LABEL_COL] = h2o_frame[LABEL_COL].asfactor()

    for c in feature_cols:
        h2o_frame[c] = h2o_frame[c].asnumeric()

    pred = model.predict(h2o_frame)
    combined = h2o_frame.cbind(pred)

    out_df = hc.asSparkFrame(combined)

    print_partition_completion(out_df, "PREDICTION-OUTPUT-READY")

    print(f"Writing prediction output to: {args.output}")
    (
        out_df
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv(args.output)
    )
    print("Prediction output write finished.")

    df.unpersist()

    try:
        h2o.cluster().shutdown(prompt=False)
    except Exception as e:
        print(f"H2O shutdown warning: {e}")

    spark.stop()


if __name__ == "__main__":
    main()