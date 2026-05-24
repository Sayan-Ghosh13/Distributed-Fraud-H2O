# Step1 : Check java version (JAVA 11 required) :
java -version
# for update to java 11 : 
sudo update-alternatives --config java 
(choose java 11 version number)
sudo update-alternatives --config javac  
(choose java 11 version number)

# Step1 : Setup Hosts (IMPORTANT)
sudo nano /etc/hosts
# put master ip address as well as hostname in both master and worker : 
(ex: 172.20.252.144 bhaskara17)

# in terminal type 
nano ~/.bashrc 
# add this in bottom (if not exits)
export SPARK_HOME=/home/sysadm/spark
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3

# Step2 : Setup Passwordless SSH (On master machine)

ssh-keygen -t rsa
ssh-copy-id sysadm@bhaskara15(worker id)

# Step3 : Configure Spark
cd downloads/distributed_fraud_h2o/conf/

# edit workers file :
nano workers

# add : 
worker hostname (ex : bhaskara15)

# edit spark_env file :
nano spark-env.sh

# edit : 
export SPARK_MASTER_HOST=172.20.xxx.xxx or hostname

# Step4 :Copy config to Spark folder (On master machine)

cd /home/sysadm/Downloads/distributed_fraud_h2o
cp conf/spark-env.sh /home/sysadm/spark/conf/
cp conf/workers /home/sysadm/spark/conf/
scp conf/spark-env.sh sysadm@bhaskara15:/home/sysadm/spark/conf/
scp conf/workers sysadm@bhaskara15:/home/sysadm/spark/conf/

# Step5 : Start master (on master machine):

cd $SPARK_HOME
sbin/start-master.sh

# Step6 : Start worker (On worker machine) :

cd $SPARK_HOME
sbin/start-worker.sh spark://bhaskara16:7077

# Step7 : check Ui : 

http://bhaskara16:8080

# Step8 : Run Spark job :

cd /home/sysadm/Downloads/distributed_fraud_h2o

spark-submit \
  --master spark://bhaskara16:7077 \
  --deploy-mode client \
  --conf spark.scheduler.minRegisteredResourcesRatio=1 \
  --jars /home/sysadm/Downloads/sparkling-water-3.46.0.6-1-3.5/jars/sparkling-water-assembly_2.12-3.46.0.6-1-3.5-all.jar \
  --py-files /home/sysadm/Downloads/sparkling-water-3.46.0.6-1-3.5/py/h2o_pysparkling_3.5-3.46.0.6-1-3.5.zip \
  src/train_fraud_h2o.py \
  --input file:///home/sysadm/Downloads/distributed_fraud_h2o/data/creditcard.csv \
  --model-dir /tmp/h2o_rf_model \
  --prediction-output /tmp/h2o_test_predictions \
  --results-dir results \
  --repartition 8 \
  --ntrees 300 \
  --max-depth 25 \
  --sample-rate 0.9 \
  --seed 42

# Step9 : Check Output :
ls /tmp/h2o_rf_model
ls /tmp/h2o_test_predictions

cat results/spark_metrics.txt
cat results/h2o_metrics.txt
cat results/comparison.txt
