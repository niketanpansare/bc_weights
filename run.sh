#!/bin/bash
export HADOOP_CONF_DIR=/usr/iop/current/hadoop-client/conf
export SPARK_HOME=/home/biuser/spark-2.1.0-bin-hadoop2.7
$SPARK_HOME/bin/spark-submit --master yarn-client  --driver-memory 20G --num-executors 6 --executor-memory 55G --executor-cores 24  --conf spark.driver.maxResultSize=0 --conf spark.akka.frameSize=128 --conf spark.yarn.executor.memoryOverhead=8250 --conf spark.network.timeout=6000s --conf spark.rpc.askTimeout=6000s --conf spark.memory.useLegacyMode=true --conf spark.files.useFetchCache=false --conf "spark.executor.extraJavaOptions=-Xmn6G -server" --conf  "spark.driver.extraJavaOptions=-Xmn2G -server" --conf "spark.ui.port=4050" SystemML.jar -f resnet_prediction_for.dml -stats
