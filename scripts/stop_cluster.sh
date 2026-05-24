#!/usr/bin/env bash
set -e

if [ -z "$SPARK_HOME" ]; then
  echo "SPARK_HOME is not set"
  exit 1
fi

echo "Stopping Spark workers..."
"$SPARK_HOME/sbin/stop-workers.sh"

echo "Stopping Spark master..."
"$SPARK_HOME/sbin/stop-master.sh"

echo "Spark cluster stopped."