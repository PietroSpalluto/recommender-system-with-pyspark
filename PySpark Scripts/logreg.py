from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler , StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from os import path
import numpy as np
import pandas as pd
import time
import bgrfunctions as bgrf

spark = SparkSession.builder.appName("boardgamesrecsys").getOrCreate()

# Parameters
main_path = '/home/sergio/spark-3.2.1-bin-hadoop3.2/esame'
n_pc = 15

complete = spark.read.csv('{}/data/clean/complete_indexed.csv'.format(main_path), inferSchema=True, header=True, sep=',')
df = spark.read.csv('{}/data/clean/user_ratings_disc.csv'.format(main_path), inferSchema=True, header=True, sep=',')
# One Hot Encoding of User Ids and Games Ids
indexer = StringIndexer(inputCol="Username", outputCol="UserId")
ohe = OneHotEncoder(inputCols=['UserId', 'BGGId'], outputCols=['UserId_onehot', 'BGGId_onehot'])

pipeline = Pipeline(stages=[indexer, ohe])
df = pipeline.fit(df).transform(df)

# The complete database and the users ratings database are joined
games_ratings = complete.join(df, 'BGGId', 'inner')


# Min Max Scaling

# The features vector does not contain rating because we want to keep it separated for the prediction
assembler = VectorAssembler(inputCols=games_ratings.drop('Username', 'Rating', 'BGGId', 'UserId', 'UserId_onehot', 'BGGId_onehot','buckets').columns, 
                                outputCol='features')

scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features')
pipeline = Pipeline(stages=[assembler, scaler])
games_ratings = pipeline.fit(games_ratings).transform(games_ratings)

pca_features = bgrf.principal_components_analysis(games_ratings, n_pc)

# Prediction using Logistic Regression
startpred=time.time()
prediction, acc, roc, best = bgrf.logistic_regression(pca_features)
endpred=time.time()

prediction.show()
print("LogisticRegression Execution time: %s min\n" %((endpred-startpred)/60))
print('Best regParam '+str(best._java_obj.parent().getRegParam()))
print('Best maxIter '+str(best._java_obj.parent().getMaxIter()))
print("Test set accuracy = " + str(acc))
print("Test set roc = " + str(roc))
spark.stop()

