from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler , StringIndexer, OneHotEncoder
from pyspark.ml.regression import FMRegressor
from pyspark.ml import Pipeline
from os import path
import numpy as np
import pandas as pd
import time
import bgrfunctions as bgrf
# Parameters
main_path = '/home/sergio/spark-3.2.1-bin-hadoop3.2/esame'
n_pc = 15

spark = SparkSession.builder.appName("boardgamesrecsys").getOrCreate()

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
assembler = VectorAssembler(inputCols=games_ratings.drop('Username', 'Rating', 'BGGId', 'UserId', 'UserId_onehot', 'BGGId_onehot').columns, 
                                outputCol='features')

scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features')
pipeline = Pipeline(stages=[assembler, scaler])
games_ratings = pipeline.fit(games_ratings).transform(games_ratings)

pca_features = bgrf.principal_components_analysis(games_ratings, n_pc)

# Prediction using FMRegressor
startpred=time.time()
prediction, rmse, r2, best = bgrf.fmreg_prediction(pca_features)
endpred=time.time()
prediction.show()
print("FM Regressor Execution time: %s min" %((endpred-startpred)/60))
print('Best StepSize '+str(best._java_obj.parent().getStepSize()))
print('Best FactorSize '+str(best._java_obj.parent().getFactorSize)))
print("rmse metric = " + str(rmse))
print("r2 metric = " + str(r2))

spark.stop()

