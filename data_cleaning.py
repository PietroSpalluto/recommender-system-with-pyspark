from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, MinMaxScaler, PCA
from pyspark.ml.regression import FMRegressor
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.stat import ChiSquareTest, Correlation

from pyspark.sql.functions import countDistinct, col, count
from pyspark.sql.types import IntegerType, BooleanType

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import functions

main_path = '/content/drive/MyDrive/Big Data/data'
spark = SparkSession.builder.master("local[*]").appName("boardgamesrecsys").getOrCreate()
numrec = 1000
numut = 10

# User ratings are read and the missing values are dropped, the result is saved in a cleaned database.
user_ratings_df = pd.read_csv('{}/user_ratings.csv'.format(main_path))
print('User ratings missing values', user_ratings_df.isna().sum().sum())
user_ratings_df = user_ratings_df.dropna()
print('User ratings missing values', user_ratings_df.isna().sum().sum())
user_ratings_df.to_csv('{}/clean/user_ratings.csv'.format(main_path), index=False)

# The cleaned database is read using spark, reading a csv is faster than converting a Pandas DataFrame in a Pyspark
# DataFrame.
user_ratings = spark.read.csv('{}/clean/user_ratings.csv'.format(main_path),
                              inferSchema=True, header=True, sep=',')

# Usernames are converted into ids and then game ids and user ids are one hot encoded. User with less than 'numut'
# entries and games with less than 'numrec' entries are removed.
df = functions.low_importance_elements(user_ratings, numrec, numut)

# One Hot Encoding of User Ids and Games Ids
indexer = StringIndexer(inputCol="Username", outputCol="UserId")
ohe = OneHotEncoder(inputCols=['UserId', 'BGGId'], outputCols=['UserId_onehot', 'BGGId_onehot'])

pipeline = Pipeline(stages=[indexer, ohe])
df2 = pipeline.fit(df).transform(df)

# df2.show()
# df2.count()

# Database are loaded
games_df = pd.read_csv('/content/drive/MyDrive/Big Data/data/games.csv')
print('Games missing values:', games_df.isna().sum().sum())
themes_df = pd.read_csv('/content/drive/MyDrive/Big Data/data/themes.csv')
print('Themes missing values:', themes_df.isna().sum().sum())
subcategories_df = pd.read_csv('/content/drive/MyDrive/Big Data/data/subcategories.csv')
print('Subcateories missing values:', subcategories_df.isna().sum().sum())
mechanics_df = pd.read_csv('/content/drive/MyDrive/Big Data/data/mechanics.csv')
print('Mechanics missing values:', mechanics_df.isna().sum().sum())
publishers_df = pd.read_csv('/content/drive/MyDrive/Big Data/data/publishers_reduced.csv')
print('Publishers missing values:', publishers_df.isna().sum().sum())
artists_df = pd.read_csv('/content/drive/MyDrive/Big Data/data/artists_reduced.csv')
print('Artists missing values:', artists_df.isna().sum().sum())
designers_df = pd.read_csv('/content/drive/MyDrive/Big Data/data/designers_reduced.csv')
print('Designers missing values:', designers_df.isna().sum().sum())
ratings_distribution_df = pd.read_csv('/content/drive/MyDrive/Big Data/data/ratings_distribution.csv')
print('Ratings missing values:', ratings_distribution_df.isna().sum().sum())

categorical_features = functions.make_categorical_db(themes_df, games_df, subcategories_df, mechanics_df, artists_df,
                                                     designers_df, publishers_df)
print('Categorical features missing values:', categorical_features.isna().sum().sum())
categorical_features.to_csv('{}/categorical_features.csv'.format(main_path), index=False)

# Games and Categorical Features DataFrames are concatenated to obtain a complete database
complete_df = pd.concat([games_df, categorical_features], axis=1)

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

complete_df.to_csv('{}/complete.csv'.format(main_path), index=False)

# Missing values plot
functions.plot_missing_values(complete_df)

# Data cleaning and saving of the complete database.
complete_df = functions.clean_complete_database(complete_df)

print('Missing values after cleaning:', complete_df.isna().sum().sum())
complete_df.to_csv('{}/clean/complete.csv'.format(main_path), index=False)

# Categorical features encoding
complete = spark.read.csv('{}/clean/complete.csv'.format(main_path), inferSchema=True, header=True, sep=',')
complete = functions.encode_categorical_features(complete)

# The complete database and the user ratings are joined to compute the correlation matrix.
games_ratings = complete.join(df2, 'BGGId', 'right')
cols = games_ratings.drop('Username', 'BGGId', 'UserId', 'UserId_onehot', 'BGGId_onehot').columns
features_names = ['Username', 'BGGId', 'UserId', 'UserId_onehot', 'BGGId_onehot', 'features']
# games_ratings.show()
# games_ratings.count()

functions.compute_correlation_matrix(games_ratings, cols, features_names)

pca_model = functions.principal_components_analysis(games_ratings)
functions.compute_heatmap(pca_model, games_ratings)

# Variance explained plot
plt.plot(range(1, 11), pca_model.explainedVariance.cumsum(), marker='o', linestyle='--')
plt.title('variance by components')
plt.xlabel('num of components')
plt.ylabel('cumulative explained variance')

pca_features = pca_model.transform(games_ratings)

# pca_features.show()
# pca_features.count()

cols = ['pca_features', 'Rating']
features_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Rating']
functions.compute_correlation_matrix(pca_features, cols, features_names)

prediction = functions.fm_prediction(pca_features)
