#!/usr/bin/env bash

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export SPARK_HOME=/home/sysadm/spark
export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop

# master machine host/IP
export SPARK_MASTER_HOST=bhaskara16
export SPARK_MASTER_PORT=7077
export SPARK_MASTER_WEBUI_PORT=8080

# worker settings
export SPARK_WORKER_CORES=3
export SPARK_WORKER_MEMORY=4g
export SPARK_WORKER_PORT=8881
export SPARK_WORKER_WEBUI_PORT=8081

# python
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3