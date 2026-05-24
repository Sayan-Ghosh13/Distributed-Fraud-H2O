#!/usr/bin/env bash
set -e

if [ -z "$SPARK_HOME" ]; then
  echo "SPARK_HOME is not set"
  exit 1
fi

echo "Starting Spark master..."
"$SPARK_HOME/sbin/start-master.sh"

echo "Starting Spark workers..."
"$SPARK_HOME/sbin/start-workers.sh"

echo "Spark cluster started."
echo "Master UI: http://$(hostname -f):8080"