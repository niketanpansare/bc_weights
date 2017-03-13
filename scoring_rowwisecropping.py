import os
import numpy as np
from pyspark.sql.functions import col, max
import systemml  # pip3 install systemml
from systemml import MLContext, dml
from pyspark.context import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext()
sqlContext = SQLContext(sc)
ml = MLContext(sc)
# train_df = sqlContext.read.load('data/train_256.parquet')
val_df = sqlContext.read.load('data/val_256.parquet')

X_val = val_df.select("__INDEX", "sample")
ml.setStatistics(True).setStatisticsMaxHeavyHitters(30).setExplain(True)
script = dml("resnet_prediction_parfor_rowwisecropping.dml").input(X=X_val).output("Y")
Y = ml.execute(script).get("Y").toDF()
Y.show()
