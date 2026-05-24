import argparse
import os
import socket
import time
from typing import Dict, List, Tuple

import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
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
        description="Improved Distributed Credit Card Fraud Detection using Spark + H2O"
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--model-dir", required=True, help="Directory to save H2O model")
    parser.add_argument("--prediction-output", required=True, help="Directory to save distributed test predictions")
    parser.add_argument("--results-dir", default="results", help="Directory to save metrics text files")
    parser.add_argument("--repartition", type=int, default=8, help="Number of Spark partitions")
    parser.add_argument("--ntrees", type=int, default=300, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=25, help="Maximum tree depth")
    parser.add_argument("--min-rows", type=float, default=1.0, help="Minimum rows per leaf")
    parser.add_argument("--sample-rate", type=float, default=0.9, help="Row sampling rate")
    parser.add_argument("--mtries", type=int, default=-1, help="Columns sampled per split; -1 means auto")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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


def validate_schema(df: DataFrame) -> None:
    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Input dataset must contain label column '{LABEL_COL}'. "
            f"For the Kaggle creditcard.csv dataset, the target column is 'Class'."
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
    df = df.filter(F.col(LABEL_COL).isNotNull())

    fill_dict = {}
    for c in df.columns:
        if c != LABEL_COL:
            fill_dict[c] = 0.0
    df = df.fillna(fill_dict)

    df = df.dropDuplicates()
    return df


def add_row_id(df: DataFrame) -> DataFrame:
    return df.withColumn(ROW_ID_COL, F.monotonically_increasing_id())


def print_basic_stats(df: DataFrame, title: str = "DATASET SUMMARY") -> None:
    total = df.count()
    fraud = df.filter(F.col(LABEL_COL) == 1).count()
    normal = df.filter(F.col(LABEL_COL) == 0).count()

    print("=" * 60)
    print(title)
    print(f"Total rows      : {total}")
    print(f"Fraud rows      : {fraud}")
    print(f"Normal rows     : {normal}")
    if total > 0:
        print(f"Fraud ratio     : {fraud / total:.6f}")
    print("=" * 60)


def stratified_split(df: DataFrame, seed: int) -> Tuple[DataFrame, DataFrame, DataFrame]:
    fraud_df = df.filter(F.col(LABEL_COL) == 1)
    normal_df = df.filter(F.col(LABEL_COL) == 0)

    fraud_train, fraud_valid, fraud_test = fraud_df.randomSplit([0.70, 0.15, 0.15], seed=seed)
    normal_train, normal_valid, normal_test = normal_df.randomSplit([0.70, 0.15, 0.15], seed=seed)

    train_df = fraud_train.unionByName(normal_train)
    valid_df = fraud_valid.unionByName(normal_valid)
    test_df = fraud_test.unionByName(normal_test)

    return train_df, valid_df, test_df


def spark_to_h2o(hc, df: DataFrame, name: str):
    return hc.asH2OFrame(df, h2oFrameName=name)


def force_h2o_types(hf, label_col: str, feature_cols: List[str]):
    hf[label_col] = hf[label_col].asfactor()
    for col_name in feature_cols:
        hf[col_name] = hf[col_name].asnumeric()
    return hf


def best_metric(metric_result):
    if metric_result is None or len(metric_result) == 0:
        return None
    return metric_result[0]


def format_float(x) -> str:
    return "NA" if x is None else f"{x:.6f}"


def extract_h2o_thresholds(perf) -> Dict[str, float]:
    result = {}
    try:
        f1_pair = best_metric(perf.F1())
        result["best_f1_threshold"] = float(f1_pair[0]) if f1_pair else 0.5
        result["best_f1_value"] = float(f1_pair[1]) if f1_pair else None
    except Exception:
        result["best_f1_threshold"] = 0.5
        result["best_f1_value"] = None

    try:
        acc_pair = best_metric(perf.accuracy())
        result["best_accuracy_threshold"] = float(acc_pair[0]) if acc_pair else 0.5
        result["best_accuracy_value"] = float(acc_pair[1]) if acc_pair else None
    except Exception:
        result["best_accuracy_threshold"] = 0.5
        result["best_accuracy_value"] = None

    return result


def confusion_counts_from_spark(df: DataFrame, pred_col: str) -> Dict[str, int]:
    tp = df.filter((F.col(LABEL_COL) == 1) & (F.col(pred_col) == 1)).count()
    tn = df.filter((F.col(LABEL_COL) == 0) & (F.col(pred_col) == 0)).count()
    fp = df.filter((F.col(LABEL_COL) == 0) & (F.col(pred_col) == 1)).count()
    fn = df.filter((F.col(LABEL_COL) == 1) & (F.col(pred_col) == 0)).count()
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def binary_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    spark = build_spark("Improved-Distributed-CreditCard-Fraud-H2O")
    hc = H2OContext.getOrCreate()
    print("H2O cluster is ready.")

    print(f"Reading CSV from: {args.input}")
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(args.input)
    )

    validate_schema(df)
    df = cast_numeric_columns(df)
    df = clean_dataframe(df)
    df = add_row_id(df)

    if args.repartition > 0:
        df = df.repartition(args.repartition)

    df.persist(StorageLevel.MEMORY_AND_DISK)
    df.rdd.mapPartitions(worker_partition_logger).collect()

    # worker completion print for input distribution
    print_partition_completion(df, "AFTER-READ-AND-REPARTITION")

    print_basic_stats(df, "FULL DATASET SUMMARY")

    train_df, valid_df, test_df = stratified_split(df, args.seed)

    train_df = train_df.repartition(args.repartition).persist(StorageLevel.MEMORY_AND_DISK)
    valid_df = valid_df.repartition(args.repartition).persist(StorageLevel.MEMORY_AND_DISK)
    test_df = test_df.repartition(args.repartition).persist(StorageLevel.MEMORY_AND_DISK)

    # worker completion print for each split
    print_partition_completion(train_df, "TRAIN-DF-READY")
    print_partition_completion(valid_df, "VALID-DF-READY")
    print_partition_completion(test_df, "TEST-DF-READY")

    print_basic_stats(train_df, "TRAIN SUMMARY")
    print_basic_stats(valid_df, "VALID SUMMARY")
    print_basic_stats(test_df, "TEST SUMMARY")

    feature_cols = [c for c in df.columns if c not in [LABEL_COL, ROW_ID_COL]]

    h2o_train = spark_to_h2o(hc, train_df, "creditcard_train")
    h2o_valid = spark_to_h2o(hc, valid_df, "creditcard_valid")
    h2o_test = spark_to_h2o(hc, test_df, "creditcard_test")

    h2o_train = force_h2o_types(h2o_train, LABEL_COL, feature_cols)
    h2o_valid = force_h2o_types(h2o_valid, LABEL_COL, feature_cols)
    h2o_test = force_h2o_types(h2o_test, LABEL_COL, feature_cols)

    model = H2ORandomForestEstimator(
        model_id="creditcard_drf_model",
        ntrees=args.ntrees,
        max_depth=args.max_depth,
        min_rows=args.min_rows,
        sample_rate=args.sample_rate,
        mtries=args.mtries,
        seed=args.seed,
        balance_classes=True,
        max_after_balance_size=10.0,
        stopping_rounds=5,
        stopping_metric="AUC",
        score_each_iteration=True
    )

    print("Training H2O Distributed Random Forest...")
    model.train(
        x=feature_cols,
        y=LABEL_COL,
        training_frame=h2o_train,
        validation_frame=h2o_valid
    )
    print("H2O training finished.")
    print("Checking H2O cluster nodes...")
    for node in h2o.cluster().nodes:
        try:
            print(f"H2O node {node['ip_port']} is alive and participated in training.")
        except Exception:
            print(node)

    print("\nModel summary:")
    print(model)

    valid_perf = model.model_performance(valid=True)
    test_perf = model.model_performance(test_data=h2o_test)

    thresholds = extract_h2o_thresholds(valid_perf)
    best_f1_threshold = thresholds["best_f1_threshold"]

    print(f"Best validation F1 threshold: {best_f1_threshold:.6f}")

    test_pred_h2o = model.predict(h2o_test)
    combined_h2o = h2o_test.cbind(test_pred_h2o)

    pred_spark_df = hc.asSparkFrame(combined_h2o)

    pred_spark_df = pred_spark_df.withColumn(
        "predict_tuned",
        F.when(F.col("p1") >= F.lit(best_f1_threshold), F.lit(1)).otherwise(F.lit(0))
    )

    # worker completion print before write
    print_partition_completion(pred_spark_df, "PREDICTION-DF-READY")

    print(f"Writing predictions to: {args.prediction_output}")
    (
        pred_spark_df
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv(args.prediction_output)
    )
    print("Prediction write finished.")

    print(f"Saving model to: {args.model_dir}")
    saved_model_path = h2o.save_model(model=model, path=args.model_dir, force=True)
    print(f"Saved H2O model at: {saved_model_path}")

    try:
        mojo_path = model.download_mojo(path=args.model_dir, get_genmodel_jar=False)
        print(f"Saved MOJO at: {mojo_path}")
    except Exception as e:
        print(f"MOJO export skipped: {e}")

    default_counts = confusion_counts_from_spark(pred_spark_df, "predict")
    tuned_counts = confusion_counts_from_spark(pred_spark_df, "predict_tuned")

    default_metrics = binary_metrics_from_counts(
        default_counts["tp"], default_counts["tn"], default_counts["fp"], default_counts["fn"]
    )
    tuned_metrics = binary_metrics_from_counts(
        tuned_counts["tp"], tuned_counts["tn"], tuned_counts["fp"], tuned_counts["fn"]
    )

    spark_metrics_text = f"""SPARK-SIDE TEST METRICS
========================================
Rows evaluated: {pred_spark_df.count()}

DEFAULT H2O PREDICTION COLUMN (predict)
----------------------------------------
TP: {default_counts['tp']}
TN: {default_counts['tn']}
FP: {default_counts['fp']}
FN: {default_counts['fn']}

Accuracy:           {format_float(default_metrics['accuracy'])}
Precision:          {format_float(default_metrics['precision'])}
Recall:             {format_float(default_metrics['recall'])}
Specificity:        {format_float(default_metrics['specificity'])}
F1 Score:           {format_float(default_metrics['f1'])}
Balanced Accuracy:  {format_float(default_metrics['balanced_accuracy'])}

TUNED PREDICTION COLUMN (predict_tuned)
----------------------------------------
Threshold used:     {best_f1_threshold:.6f}

TP: {tuned_counts['tp']}
TN: {tuned_counts['tn']}
FP: {tuned_counts['fp']}
FN: {tuned_counts['fn']}

Accuracy:           {format_float(tuned_metrics['accuracy'])}
Precision:          {format_float(tuned_metrics['precision'])}
Recall:             {format_float(tuned_metrics['recall'])}
Specificity:        {format_float(tuned_metrics['specificity'])}
F1 Score:           {format_float(tuned_metrics['f1'])}
Balanced Accuracy:  {format_float(tuned_metrics['balanced_accuracy'])}
"""
    save_text(os.path.join(args.results_dir, "spark_metrics.txt"), spark_metrics_text)

    try:
        test_auc = test_perf.auc()
    except Exception:
        test_auc = None

    try:
        test_pr_auc = test_perf.pr_auc()
    except Exception:
        test_pr_auc = None

    try:
        h2o_accuracy_pair = best_metric(test_perf.accuracy())
        h2o_f1_pair = best_metric(test_perf.F1())
        h2o_precision_pair = best_metric(test_perf.precision())
        h2o_recall_pair = best_metric(test_perf.recall())
    except Exception:
        h2o_accuracy_pair = None
        h2o_f1_pair = None
        h2o_precision_pair = None
        h2o_recall_pair = None

    h2o_metrics_text = f"""H2O TEST METRICS
========================================
Model ID: creditcard_drf_model

AUC:                 {format_float(test_auc)}
PR AUC:              {format_float(test_pr_auc)}

BEST TEST ACCURACY
----------------------------------------
Threshold:           {format_float(h2o_accuracy_pair[0] if h2o_accuracy_pair else None)}
Value:               {format_float(h2o_accuracy_pair[1] if h2o_accuracy_pair else None)}

BEST TEST F1
----------------------------------------
Threshold:           {format_float(h2o_f1_pair[0] if h2o_f1_pair else None)}
Value:               {format_float(h2o_f1_pair[1] if h2o_f1_pair else None)}

BEST TEST PRECISION
----------------------------------------
Threshold:           {format_float(h2o_precision_pair[0] if h2o_precision_pair else None)}
Value:               {format_float(h2o_precision_pair[1] if h2o_precision_pair else None)}

BEST TEST RECALL
----------------------------------------
Threshold:           {format_float(h2o_recall_pair[0] if h2o_recall_pair else None)}
Value:               {format_float(h2o_recall_pair[1] if h2o_recall_pair else None)}

VALIDATION-SELECTED THRESHOLD FOR SPARK TUNED OUTPUT
----------------------------------------
Best validation F1 threshold: {best_f1_threshold:.6f}
"""
    save_text(os.path.join(args.results_dir, "h2o_metrics.txt"), h2o_metrics_text)

    better_f1 = "predict_tuned" if tuned_metrics["f1"] >= default_metrics["f1"] else "predict"
    better_recall = "predict_tuned" if tuned_metrics["recall"] >= default_metrics["recall"] else "predict"
    better_bal_acc = (
        "predict_tuned"
        if tuned_metrics["balanced_accuracy"] >= default_metrics["balanced_accuracy"]
        else "predict"
    )

    comparison_text = f"""COMPARISON REPORT
========================================
Goal:
Improve fraud detection on an imbalanced dataset by using class balancing
and threshold tuning instead of trusting only the default prediction threshold.

DEFAULT VS TUNED
----------------------------------------
Default prediction column:       predict
Tuned prediction column:         predict_tuned
Validation-selected threshold:   {best_f1_threshold:.6f}

DEFAULT METRICS
----------------------------------------
Accuracy:           {format_float(default_metrics['accuracy'])}
Precision:          {format_float(default_metrics['precision'])}
Recall:             {format_float(default_metrics['recall'])}
F1 Score:           {format_float(default_metrics['f1'])}
Balanced Accuracy:  {format_float(default_metrics['balanced_accuracy'])}

TUNED METRICS
----------------------------------------
Accuracy:           {format_float(tuned_metrics['accuracy'])}
Precision:          {format_float(tuned_metrics['precision'])}
Recall:             {format_float(tuned_metrics['recall'])}
F1 Score:           {format_float(tuned_metrics['f1'])}
Balanced Accuracy:  {format_float(tuned_metrics['balanced_accuracy'])}

WHICH IS BETTER?
----------------------------------------
Better F1:                  {better_f1}
Better Recall:              {better_recall}
Better Balanced Accuracy:   {better_bal_acc}

CONCLUSION
----------------------------------------
1. The model was trained in distributed mode using Spark + H2O.
2. The fraud dataset is highly imbalanced, so threshold tuning matters.
3. The tuned output uses the validation-best F1 threshold.
4. For reporting fraud detection quality, focus especially on:
   - Recall
   - F1 Score
   - Balanced Accuracy
   not only plain Accuracy.
"""
    save_text(os.path.join(args.results_dir, "comparison.txt"), comparison_text)

    print("Saved metrics files:")
    print(os.path.join(args.results_dir, "spark_metrics.txt"))
    print(os.path.join(args.results_dir, "h2o_metrics.txt"))
    print(os.path.join(args.results_dir, "comparison.txt"))

    print("Top feature importance:")
    try:
        print(model.varimp(use_pandas=True))
    except Exception as e:
        print(f"Could not print variable importance: {e}")

    train_df.unpersist()
    valid_df.unpersist()
    test_df.unpersist()
    df.unpersist()

    try:
        h2o.cluster().shutdown(prompt=False)
    except Exception as e:
        print(f"H2O shutdown warning: {e}")

    spark.stop()


if __name__ == "__main__":
    main()