import numpy as np
import pandas as pd

from pyspark.ml.feature import StringIndexer, VectorAssembler, PCA
from pyspark.ml.stat import ChiSquareTest, Correlation
from pyspark.ml.regression import FMRegressor

from pyspark.sql.functions import countDistinct, col, count
from pyspark.sql.types import IntegerType, BooleanType

from matplotlib import pyplot as plt


def find_most_frequent_value(dataframe, feature):
    features = dataframe.groupby(by='{}'.format(feature))
    features = {key: len(value) for key, value in features.groups.items()}
    features = pd.DataFrame(data=features.values(), index=features.keys())
    features.sort_values(by=0, ascending=False, inplace=True)
    return features.iloc[0].name


def remove_points(dfs, names):
    for df, name in zip(dfs, names):
        d = {}
        for column in df.columns:
            d[column] = column.replace('.', '')
        df.rename(columns=d, inplace=True)
        df.to_csv('/content/drive/MyDrive/Big Data/data/clean/{}.csv'.format(name))


def binary_to_categorical(cat_features, cat_df):
    for game in cat_df:
        cat_list = [cat for cat in cat_df.columns if cat.iloc[game][cat] != 0 and cat != 'BGGId']
        cat_features.append(', '.join(cat_list))

    return cat_features


def low_importance_elements(df, numrec, numut):

    df1 = df.groupBy("BGGId").count()
    df1 = df1.filter(df1["count"] > numrec)
    df1 = df.join(df1, 'BGGId', "leftsemi")

    df2 = df.groupBy("Username").count()
    df2 = df2.filter(df2["count"] > numut)
    df2 = df.join(df2, 'Username', "leftsemi")

    return df1.join(df2, 'Username', "leftsemi")


def make_categorical_db(themes_df, games_df, subcategories_df, mechanics_df, artists_df, designers_df, publishers_df):
    # The databases containing the binary representation of the categorical features are used to create a database
    # with a smaller dimensionality, useful for visualization purposes.
    themes = []
    mechanics = []
    categories = []
    subcategories = []
    artists = []
    publishers = []
    designers = []

    categorical_features = pd.DataFrame()
    categorical_features['Themes'] = binary_to_categorical(themes, themes_df)
    categorical_features['Categories'] = binary_to_categorical(categories, games_df[games_df.columns[40:48]])
    categorical_features['Subcategories'] = binary_to_categorical(subcategories, subcategories_df)
    categorical_features['Mechanics'] = binary_to_categorical(mechanics, mechanics_df)
    categorical_features['Artists'] = binary_to_categorical(artists, artists_df)
    categorical_features['Designers'] = binary_to_categorical(designers, designers_df)
    categorical_features['Publishers'] = binary_to_categorical(publishers, publishers_df)

    return categorical_features


def plot_missing_values(complete_df):
    plt.figure(figsize=(8, 10))
    plt.barh(complete_df.columns, complete_df.isna().sum())
    plt.title('Missing Values')
    plt.yticks(range(len(complete_df.columns)), complete_df.columns, rotation=00)
    plt.gca().xaxis.grid(True)


def clean_complete_database(complete_df):
    # Some features are dropped because they are not used in the analysis
    complete_df.drop('Name', axis=1, inplace=True)
    complete_df.drop('ImagePath', axis=1, inplace=True)
    complete_df.drop('Description', axis=1, inplace=True)
    complete_df.drop('GoodPlayers', axis=1, inplace=True)
    complete_df.drop('BestPlayers', axis=1, inplace=True)
    complete_df.drop('NumComments', axis=1, inplace=True)  # This is always 0

    # Family has missing values that can represent a game that doesn't belong to a family of games
    complete_df['Family'].fillna('No family', inplace=True)

    # Missing values in ComAgeRec and LanguageEase are replaced with the mean value
    complete_df['ComAgeRec'].fillna(complete_df['ComAgeRec'].mean(), inplace=True)
    complete_df['LanguageEase'].fillna(complete_df['ComAgeRec'].mean(), inplace=True)

    # The missing values in Themes, Mechanics, Designers and Publishers are replaced with the most frequent value
    most_frequent_theme = find_most_frequent_value(complete_df, 'Themes')
    complete_df['Themes'].fillna(most_frequent_theme, inplace=True)

    most_frequent_mechanic = find_most_frequent_value(complete_df, 'Mechanics')
    complete_df['Mechanics'].fillna(most_frequent_mechanic, inplace=True)

    most_frequent_publisher = find_most_frequent_value(complete_df, 'Publishers')
    complete_df['Publishers'].fillna(most_frequent_publisher, inplace=True)

    most_frequent_designer = find_most_frequent_value(complete_df, 'Designers')
    complete_df['Designers'].fillna(most_frequent_designer, inplace=True)

    # Artists, Categories and Subcategories are dropped
    complete_df.drop('Artists', axis=1, inplace=True)
    complete_df.drop('Categories', axis=1, inplace=True)
    complete_df.drop('Subcategories', axis=1, inplace=True)

    return complete_df


def encode_categorical_features(complete):
    # Some columns are converted to integer or boolean
    complete = complete.withColumn('NumOwned', complete['NumOwned'].cast(IntegerType()))
    complete = complete.withColumn('Kickstarted', complete['Kickstarted'].cast(BooleanType()))
    complete = complete.withColumn('Rank:boardgame', complete['Rank:boardgame'].cast(IntegerType()))
    complete = complete.withColumn('YearPublished', complete['YearPublished'].cast(IntegerType()))

    # Categorical features encoding
    stringIndexer = StringIndexer(inputCol="Themes", outputCol="Themes_indexed")
    complete_indexed = stringIndexer.fit(complete).transform(complete)
    stringIndexer = StringIndexer(inputCol="Mechanics", outputCol="Mechanics_indexed")
    complete_indexed = stringIndexer.fit(complete_indexed).transform(complete_indexed)
    stringIndexer = StringIndexer(inputCol="Designers", outputCol="Designers_indexed")
    complete_indexed = stringIndexer.fit(complete_indexed).transform(complete_indexed)
    stringIndexer = StringIndexer(inputCol="Publishers", outputCol="Publishers_indexed")
    complete_indexed = stringIndexer.fit(complete_indexed).transform(complete_indexed)
    stringIndexer = StringIndexer(inputCol="Family", outputCol="Family_indexed")
    complete_indexed = stringIndexer.fit(complete_indexed).transform(complete_indexed)

    # Columns representing the categorical features are removed
    complete_indexed = complete_indexed.drop('Themes')
    complete_indexed = complete_indexed.drop('Mechanics')
    complete_indexed = complete_indexed.drop('Designers')
    complete_indexed = complete_indexed.drop('Publishers')
    complete_indexed = complete_indexed.drop('Family')

    # complete_indexed.coalesce(1).write.mode('overwrite').csv('/content/drive/MyDrive/Big_Data/data/clean/complete_indexed.csv')
    # complete_indexed.show()
    # complete_indexed.count()

    return complete_indexed


def compute_correlation_matrix(games_ratings, cols, features_names):
    assembler = VectorAssembler(
        inputCols=cols,
        outputCol='features')
    games_ratings = assembler.transform(games_ratings)

    correlation_matrix = Correlation.corr(games_ratings, 'features').collect()[0][0].toArray()

    plt.figure(figsize=(15, 15))
    plt.imshow(correlation_matrix, cmap='RdYlBu')
    plt.xticks(range(len(features_names)), features_names, rotation=90)
    plt.yticks(range(len(features_names)), features_names)
    plt.colorbar()
    print(col)

    # Correlation of all the features with the label (Rating)
    n_features = len(features_names)
    data_frame = pd.DataFrame({'Feature': features_names, 'Pearson Coefficient': correlation_matrix[n_features-1][:]})
    print(data_frame.sort_values(by='Pearson Coefficient', ascending=False))


def principal_components_analysis(games_ratings):
    # The features vector does not contain rating because we want to keep it separated for the prediction
    assembler = VectorAssembler(
        inputCols=games_ratings.drop('Username', 'Rating', 'BGGId', 'UserId', 'UserId_onehot', 'BGGId_onehot').columns,
        outputCol='features')
    games_ratings = assembler.transform(games_ratings)

    pca = PCA(k=10, inputCol='features', outputCol='pca_features')

    pca_model = pca.fit(games_ratings)

    return pca_model


# Prediction using Factorization Machine
def fm_prediction(pca_features):
    # Prediction using Factorization Machine
    pca_features = pca_features.select('UserId_onehot', 'BGGId_onehot', 'pca_features', 'Rating')

    assembler = VectorAssembler(inputCols=pca_features.drop('Rating').columns, outputCol='transformed_features')
    pca_features_rating = assembler.transform(pca_features)

    fm = FMRegressor(featuresCol="transformed_features", labelCol='Rating', stepSize=0.01, factorSize=2)

    (training, test) = pca_features_rating.randomSplit([0.8, 0.2])

    model = fm.fit(training)

    prediction = model.transform(test)
    # prediction.show()
    # prediction.count()

    return prediction


def compute_heatmap(pca_model, games_ratings):
    plt.figure(figsize=(15, 15))
    plt.imshow(pca_model.pc.toArray(), cmap='RdYlBu')
    plt.colorbar()
    cols = games_ratings.drop('Username', 'BGGId', 'Rating', 'UserId', 'UserId_onehot', 'BGGId_onehot').columns
    plt.yticks(range(len(cols)), cols)
    plt.xticks(range(0, 10))
