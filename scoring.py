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

def preprocess(df):
  X_df = df.select("__INDEX", "sample")
  y_df = df.select("__INDEX", "tumor_score")
  script = """
  crop_rgb = function(matrix[double] input, int Hin, int Win, int Hout, int Wout) return (matrix[double] out) {
    start_h = ceil((Hin - Hout) / 2)
    end_h = start_h + Hout - 1
    start_w = ceil((Win - Wout) / 2)
    end_w = start_w + Wout - 1
    mask = matrix(0, rows=Hin, cols=Win)
    temp_mask = matrix(1, rows=Hout, cols=Wout)
    mask[start_h:end_h, start_w:end_w] = temp_mask
    mask = matrix(mask, rows=1, cols=Hin*Win)
    mask = cbind(cbind(mask, mask), mask)
    out = removeEmpty(target=(input+1), margin="cols", select=mask) - 1
  }
  X = crop_rgb(X, 256, 256, 224, 224)
  # Scale images to [-1,1]
  X = X / 255
  X = X * 2 - 1
  # One-hot encode the labels
  num_tumor_classes = 3
  n = nrow(y)
  Y = table(seq(1, n), y, n, num_tumor_classes)
  """
  outputs = ("X", "Y")
  script = dml(script).input(X=X_df, y=y_df).output(*outputs)
  X, Y = ml.execute(script).get(*outputs)
  return X, Y

X_val, Y_val = preprocess(val_df)

ml.setStatistics(True).setStatisticsMaxHeavyHitters(30).setExplain(True)
script = dml("resnet_prediction_parfor.dml").input(X=X_val).output("Y")
Y = ml.execute(script).get("Y").toDF()
Y.show()
