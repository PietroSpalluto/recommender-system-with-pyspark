from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from os import path
import numpy as np
import pandas as pd
import time
import bgrfunctions as bgrf

main_path = '/home/sergio/spark-3.2.1-bin-hadoop3.2/esame'

spark=SparkSession.builder.appName("boardgamesrecsys-data-cleaning").getOrCreate()
start_time=time.time()
# User ratings are read and the missing values are dropped, the result is saved in a cleaned database
user_ratings_df = pd.read_csv('{}/data/user_ratings.csv'.format(main_path))
print('User ratings missing values:', user_ratings_df.isna().sum().sum())
user_ratings_df = user_ratings_df.dropna()
user_ratings_df = bgrf.round_ratings(user_ratings_df)
print('User ratings missing values after cleaning:', user_ratings_df.isna().sum().sum())
user_ratings_df.to_csv('{}/data/clean/user_ratings.csv'.format(main_path), index=False)

del user_ratings_df

# The cleaned database is read using spark, reading a csv is faster than converting a Pandas DataFrame in a Pyspark DataFrame
print('reading ratings...')
df = spark.read.csv('{}/data/clean/user_ratings.csv'.format(main_path), inferSchema=True, header=True, sep=',')


df = bgrf.discretize_ratings(df)
df.coalesce(1).write.option('header', True).mode('overwrite').csv('{}/data/clean/user_ratings_disc.csv'.format(main_path))
# Databases are loaded
games_df = pd.read_csv('{}/data/games.csv'.format(main_path))
print('Games missing values:', games_df.isna().sum().sum())
themes_df = pd.read_csv('{}/data/themes.csv'.format(main_path))
print('Themes missing values:', themes_df.isna().sum().sum())
subcategories_df = pd.read_csv('{}/data/subcategories.csv'.format(main_path))
print('Subcateories missing values:', subcategories_df.isna().sum().sum())
mechanics_df = pd.read_csv('{}/data/mechanics.csv'.format(main_path))
print('Mechanics missing values:', mechanics_df.isna().sum().sum())
publishers_df = pd.read_csv('{}/data/publishers_reduced.csv'.format(main_path))
print('Publishers missing values:', publishers_df.isna().sum().sum())
artists_df = pd.read_csv('{}/data/artists_reduced.csv'.format(main_path))
print('Artists missing values:', artists_df.isna().sum().sum())
designers_df = pd.read_csv('{}/data/designers_reduced.csv'.format(main_path))
print('Designers missing values:', designers_df.isna().sum().sum())
ratings_distribution_df = pd.read_csv('{}/data/ratings_distribution.csv'.format(main_path))
print('Ratings missing values:', ratings_distribution_df.isna().sum().sum())

# the dataframe representing the categorical features is created or read if it already exists
cf_path = '{}/data/categorical_features.csv'.format(main_path)
#if not path.exists(cf_path):
#    categorical_features = bgrf.make_categorical_db(themes_df, games_df, subcategories_df, mechanics_df, artists_df, designers_df, publishers_df)
#    categorical_features.to_csv('{}/data/categorical_features.csv'.format(main_path), index=False)
#else:
categorical_features = pd.read_csv('{}/data/categorical_features.csv'.format(main_path))

print('Categorical features missing values:', categorical_features.isna().sum().sum())

# Games and Categorical Features DataFrames are concatenated to obtain a complete database
print('complete database...')
complete_df = pd.concat([games_df, categorical_features], axis=1)

del games_df, themes_df, subcategories_df, mechanics_df, publishers_df, artists_df, designers_df, ratings_distribution_df, categorical_features

# Categories column are dropped because there is a categorical feature to represent them
complete_df.drop('Cat:Thematic', axis=1, inplace=True)
complete_df.drop('Cat:Strategy', axis=1, inplace=True)
complete_df.drop('Cat:War', axis=1, inplace=True)
complete_df.drop('Cat:Family', axis=1, inplace=True)
complete_df.drop('Cat:CGS', axis=1, inplace=True)
complete_df.drop('Cat:Abstract', axis=1, inplace=True)
complete_df.drop('Cat:Party', axis=1, inplace=True)
complete_df.drop('Cat:Childrens', axis=1, inplace=True)

# The column GoodPlayers contains many empty vectors that can be considered as missing values, so they are replaced
# with NaN. The same is true for BestPlayers
complete_df.loc[complete_df['GoodPlayers'] == '[]', 'GoodPlayers'] = np.nan
complete_df.loc[complete_df['BestPlayers'] == 0, 'BestPlayers'] = np.nan

print('Complete database missing values:', complete_df.isna().sum().sum())

complete_df.to_csv('{}/data/complete.csv'.format(main_path), index=False)

# Data cleaning and saving of the complete database
complete_df = bgrf.clean_complete_database(complete_df)
print('Missing values after cleaning:', complete_df.isna().sum().sum())

complete_df.to_csv('{}/data/clean/complete.csv'.format(main_path), index=False)


del complete_df

# The complete database is read as a pyspark dataframe and the categorical features are encoded using the StringIndexer
print('reading complete...')
complete = spark.read.csv('{}/data/clean/complete.csv'.format(main_path), inferSchema=True, header=True, sep=',')
complete = bgrf.encode_categorical_features(complete)

complete.coalesce(1).write.option('header', True).mode('overwrite').csv('{}/data/clean/complete_indexed.csv'.format(main_path))
print("Execution time for data cleaning: %s min" %((time.time()-start_time)/60))

spark.stop()

