from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, round
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql.types import IntegerType
import random
import time
import numpy as np
import bgrfunctions as bgrf
start_time=time.time()
numrec=1000 #minimo numero di recensioni per gioco
numut=10 #minimo numero di recensioni per utenti
pathIN='/home/sergio/spark-3.2.1-bin-hadoop3.2/esame/input/primo/user_ratings.csv'
pathINgames='/home/sergio/spark-3.2.1-bin-hadoop3.2/esame/input/primo/games.csv'

#Creazione della sessione e acquisizione del dataframe
spark=SparkSession.builder.appName('boardgamesrecsys').getOrCreate()
df=spark.read.options(delimiter=',',header=True,inferSchema=True).csv(pathIN)
dfgames=spark.read.options(delimiter=',',header=True,inferSchema=True).csv(pathINgames)

#Trasformazione Username da Stringa a indici numerici(UserId)
indexer=StringIndexer(inputCol='Username',outputCol='UserId')
df=indexer.fit(df).transform(df)

df1 = bgrf.low_importance_elements(df, numrec, numut);
df1=df1.withColumn('Rating', round(df['Rating'], 1))


startpred=time.time()
prediction, rmse, r2, best = bgrf.als_prediction(df1)
endpred=time.time()
#Valutazione del modello e parametri scelti per la creazione del modello
print('rmse metric ' + str(rmse))
print('r2 metric = ' + str(r2))
print('Best rank '+str(best.rank))
print('Best regParam '+str(best._java_obj.parent().getRegParam()))
print('Best maxIter '+str(best._java_obj.parent().getMaxIter()))

# Generate top 10 games recommendations for each user

userRecs = best.recommendForAllUsers(10)
userRecs = userRecs\
    .withColumn('rec', explode('recommendations'))\
    .select('UserId', col('rec.BGGId'), col('rec.Rating'))
dfgames=dfgames.select('BGGId', 'Name')
userRecs=userRecs.join(dfgames,['BGGId'])
userRecs=userRecs.withColumn('Rating', round(userRecs['Rating'], 1))
print('Recommendation of a random user')
userRecs.filter(userRecs['UserId']==random.randint(1, df.select('UserId').distinct().count())).show()
print('ALS Execution time: %s min' %((endpred-startpred)/60))
print('Execution time: %s min' %((time.time()-start_time)/60))

spark.stop()

#NUMERO DI UTENTI=411375
#NUMERO DI GIOCHI=21925
#NUMERO DI RIGHE=18942215
