from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, PCA
from pyspark.ml.recommendation import ALS
from pyspark.ml.regression import FMRegressor
from pyspark.ml.classification import FMClassifier, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit,TrainValidationSplitModel, ParamGridBuilder
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col, count, when, lit
from pyspark.sql.types import IntegerType, BooleanType 
from scipy import stats
from os import path
import numpy as np
import pandas as pd

def round_ratings(df):
    df['Rating'] = round(df['Rating'], 1)

    return df


def discretize_ratings(df):

    return df.withColumn('buckets', when((df['Rating'] < 4), lit(0.0)).otherwise(lit(1.0)))

    
def find_most_frequent_value(dataframe, feature):
    features = dataframe.groupby(by='{}'.format(feature))
    features = {key: len(value) for key, value in features.groups.items()}
    features = pd.DataFrame(data=features.values(), index=features.keys())
    features.sort_values(by=0, ascending=False, inplace=True)
    return features.iloc[0].name


def binary_to_categorical(cat_features, cat_df):
    for game in range(0,len(cat_df)):
        cat_list = [cat for cat in cat_df.columns if cat_df.iloc[game][cat] != 0 and cat != 'BGGId']
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


def clean_complete_database(complete_df):
    # Some features are dropped because they are not used in the analysis
    complete_df.drop('Name', axis=1, inplace=True)
    complete_df.drop('ImagePath', axis=1, inplace=True)
    complete_df.drop('Description', axis=1, inplace=True)
    complete_df.drop('GoodPlayers', axis=1, inplace=True)
    complete_df.drop('BestPlayers', axis=1, inplace=True)
    complete_df.drop('NumComments', axis=1, inplace=True)  # This is always 0
    # These features have the same values after the outliers removal
    complete_df.drop('Rank:strategygames', axis=1, inplace=True)
    complete_df.drop('Rank:abstracts', axis=1, inplace=True)
    complete_df.drop('Rank:familygames', axis=1, inplace=True)
    complete_df.drop('Rank:thematic', axis=1, inplace=True)
    complete_df.drop('Rank:cgs', axis=1, inplace=True)
    complete_df.drop('Rank:wargames', axis=1, inplace=True)
    complete_df.drop('Rank:partygames', axis=1, inplace=True)
    complete_df.drop('Rank:childrensgames', axis=1, inplace=True)

    # Incorrect values
    complete_df = complete_df[complete_df['YearPublished']>0]
    complete_df = complete_df[complete_df['MinPlayers']>0]
    complete_df = complete_df[complete_df['MaxPlayers']>0]
    complete_df = complete_df[complete_df['MfgPlaytime']>0]
    complete_df = complete_df[complete_df['ComMinPlaytime']>0]
    complete_df = complete_df[complete_df['ComMaxPlaytime']>0]
    complete_df = complete_df[complete_df['MfgAgeRec']>0]

    # Family has missing values that can represent a game that doesn't belong to a family of games
    complete_df['Family'].fillna('No family', inplace=True)

    # Missing values in ComAgeRec and LanguageEase are replaced with the mean value
    complete_df['ComAgeRec'].fillna(complete_df['ComAgeRec'].mean(), inplace=True)
    complete_df['LanguageEase'].fillna(complete_df['LanguageEase'].mean(), inplace=True)

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

    complete_df = remove_outliers(complete_df)

    # Features that have the same values (std=0 or very small) are removed
    mask = [complete_df.std()<=1e-10]

    drop_cols = complete_df.select_dtypes('number').columns[mask]
    for col in drop_cols:
        complete_df.drop(col, axis=1, inplace=True)

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

    return complete_indexed


def principal_components_analysis(games_ratings, n_pc):

    pca = PCA(k=n_pc, inputCol='scaled_features', outputCol='pca_features')
    pca_model = pca.fit(games_ratings)

    return pca_model.transform(games_ratings)
    
def als_prediction(df):
    (training, test) = df.randomSplit([0.8, 0.2])
    
    als = ALS(userCol='UserId', itemCol='BGGId', ratingCol='Rating', coldStartStrategy='drop', maxIter=20, seed=1)
    evaluator = RegressionEvaluator(metricName='rmse',labelCol='Rating', predictionCol='prediction')
    evaluator2 = RegressionEvaluator(metricName='r2',labelCol='Rating', predictionCol='prediction')

    paramGrid = ParamGridBuilder()\
        .addGrid(als.rank, [20, 30]) \
        .addGrid(als.regParam, [0.1, 0.01]) \
        .build()
	   
    tvs = TrainValidationSplit(estimator=als,estimatorParamMaps=paramGrid,evaluator=evaluator,trainRatio=0.8) 
        			   
    model = tvs.fit(training)
    
    prediction = model.transform(test)
    
    best=model.bestModel
    rmse = evaluator.evaluate(prediction)
    r2 = evaluator2.evaluate(prediction)
    
    return prediction, rmse, r2, best

# Prediction using Factorization Machine
def fmreg_prediction(pca_features):
    
    # Prediction using Factorization Machine
    model_path = '/home/sergio/spark-3.2.1-bin-hadoop3.2/esame/fmreg_model'
    step_size = [0.1, 0.01]
    factor_size = [1, 2]
    pca_features = pca_features.select('UserId', 'BGGId','UserId_onehot', 'BGGId_onehot', 'pca_features', 'Rating')

    assembler = VectorAssembler(inputCols=pca_features.drop('Rating', 'UserId', 'BGGId').columns, outputCol='transformed_features')
    pca_features_rating = assembler.transform(pca_features)

    fm = FMRegressor(featuresCol="transformed_features", labelCol='Rating', seed=1)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="Rating", predictionCol="prediction")
    evaluator2 = RegressionEvaluator(metricName="r2", labelCol="Rating", predictionCol="prediction")
    
    paramGrid = ParamGridBuilder()\
    .addGrid(fm.stepSize, [step_size[0], step_size[1]])\
    .addGrid(fm.factorSize, [factor_size[0],factor_size[1]])\
    .build()
    
    tvs = TrainValidationSplit(estimator=fm,estimatorParamMaps=paramGrid,evaluator=evaluator,trainRatio=0.8)
    (training, test) = pca_features_rating.randomSplit([0.8, 0.2])

    model = tvs.fit(training)
    model.save(model_path)
    best = model.bestModel
    prediction = model.transform(test)
    prediction=prediction.select('UserId', 'BGGId','Rating','prediction')
    rmse = evaluator.evaluate(prediction)
    r2 = evaluator2.evaluate(prediction)
    
    
    return prediction, rmse, r2, best

def fmclas_prediction(pca_features):
    # Prediction using Factorization Machine
    model_path = '/home/sergio/spark-3.2.1-bin-hadoop3.2/esame/fmclas_model'
    step_size = [0.1, 0.01]
    factor_size = [1, 2]
    pca_features = pca_features.select('UserId', 'BGGId', 'UserId_onehot', 'BGGId_onehot', 'pca_features', 'buckets')

    assembler = VectorAssembler(inputCols=pca_features.drop('UserId', 'BGGId', 'buckets').columns, outputCol='transformed_features')
    pca_features_rating = assembler.transform(pca_features)

    fm = FMClassifier(featuresCol="transformed_features", labelCol='buckets', seed=1)
    evaluator = MulticlassClassificationEvaluator(labelCol="buckets", predictionCol="prediction", metricName="accuracy")
    evaluator2 = BinaryClassificationEvaluator(labelCol="buckets", rawPredictionCol="prediction", metricName='areaUnderROC')
    
    
    paramGrid = ParamGridBuilder()\
    .addGrid(fm.stepSize, [step_size[0], step_size[1]])\
    .addGrid(fm.factorSize, [factor_size[0],factor_size[1]])\
    .build()
    
    tvs = TrainValidationSplit(estimator=fm,estimatorParamMaps=paramGrid,evaluator=evaluator,trainRatio=0.8)
    (training, test) = pca_features_rating.randomSplit([0.8, 0.2])

    model = tvs.fit(training)
    model.save(model_path)
    best = model.bestModel
    prediction = model.transform(test)
    acc = evaluator.evaluate(prediction)
    roc = evaluator2.evaluate(prediction)
    
    
    return prediction, acc, roc, best

def logistic_regression(pca_features):
    model_path = '/home/sergio/spark-3.2.1-bin-hadoop3.2/esame/logreg_model'
    reg_param = [0.1, 0.01]
    max_iter = [50, 100]
    pca_features = pca_features.select('UserId', 'BGGId', 'UserId_onehot', 'BGGId_onehot', 'pca_features', 'buckets')

    assembler = VectorAssembler(inputCols=pca_features.drop('UserId', 'BGGId', 'buckets').columns, outputCol='transformed_features')
    pca_features_rating = assembler.transform(pca_features)

    reg = LogisticRegression(featuresCol="transformed_features", labelCol='buckets')
    evaluator = MulticlassClassificationEvaluator(labelCol="buckets", predictionCol="prediction", metricName="accuracy")
    evaluator2 = BinaryClassificationEvaluator(labelCol="buckets", rawPredictionCol="prediction", metricName='areaUnderROC')
    
    
    
    paramGrid = ParamGridBuilder()\
    .addGrid(reg.regParam, [reg_param[0], reg_param[1]])\
    .addGrid(reg.maxIter, [max_iter[0], max_iter[1]])\
    .build()
    
    tvs = TrainValidationSplit(estimator=reg,estimatorParamMaps=paramGrid,evaluator=evaluator2,trainRatio=0.8)
    (training, test) = pca_features_rating.randomSplit([0.8, 0.2])

    model = tvs.fit(training)

    prediction = model.transform(test)
    best = model.bestModel
    model.save(model_path)
    
    acc = evaluator.evaluate(prediction)
    roc = evaluator2.evaluate(prediction)
    
    return prediction, acc, roc, best


def decision_tree_class(pca_features):
    model_path = '/home/sergio/spark-3.2.1-bin-hadoop3.2/esame/dectree_model'
    max_depth = [5, 10]
    min_info_gain = [20, 30]
    pca_features = pca_features.select('UserId', 'BGGId', 'UserId_onehot', 'BGGId_onehot', 'pca_features', 'buckets')

    assembler = VectorAssembler(inputCols=pca_features.drop('UserId', 'BGGId', 'buckets').columns, outputCol='transformed_features')
    pca_features_rating = assembler.transform(pca_features)

    tree = DecisionTreeClassifier(featuresCol="transformed_features", labelCol='buckets')
    evaluator = MulticlassClassificationEvaluator(labelCol="buckets", predictionCol="prediction", metricName="accuracy")
    evaluator2 = BinaryClassificationEvaluator(labelCol="buckets", rawPredictionCol="prediction", metricName='areaUnderROC')
    
    paramGrid = ParamGridBuilder()\
    .addGrid(tree.maxDepth, [max_depth[0], max_depth[1]])\
    .addGrid(tree.minInfoGain, [min_info_gain[0], min_info_gain[1]])\
    .build()
    tvs = TrainValidationSplit(estimator=tree,estimatorParamMaps=paramGrid,evaluator=evaluator2,trainRatio=0.8)
	
    (training, test) = pca_features_rating.randomSplit([0.8, 0.2])

    model = tvs.fit(training)
    model.save(model_path)
    best = model.bestModel
    prediction = model.transform(test)
    acc = evaluator.evaluate(prediction)
    roc = evaluator2.evaluate(prediction)
    
    return prediction, acc, roc, best


def random_forest_classifier(pca_features):
    model_path = '/home/sergio/spark-3.2.1-bin-hadoop3.2/esame/randforest_model'
    max_depth = [5, 10]
    min_info_gain = [20, 30]
    num_trees = 30
    pca_features = pca_features.select('UserId', 'BGGId', 'UserId_onehot', 'BGGId_onehot', 'pca_features', 'buckets')

    assembler = VectorAssembler(inputCols=pca_features.drop('UserId', 'BGGId', 'buckets').columns, outputCol='transformed_features')
    pca_features_rating = assembler.transform(pca_features)
    
    forest = RandomForestClassifier(featuresCol="transformed_features", labelCol='buckets', numTrees=num_trees)
    evaluator = MulticlassClassificationEvaluator(labelCol="buckets", predictionCol="prediction", metricName="accuracy")
    evaluator2 = BinaryClassificationEvaluator(labelCol="buckets", rawPredictionCol="prediction", metricName='areaUnderROC')
    
    
    paramGrid = ParamGridBuilder()\
    .addGrid(forest.maxDepth, [max_depth[0], max_depth[1]])\
    .addGrid(forest.minInfoGain, [min_info_gain[0], min_info_gain[1]])\
    .build()
    tvs = TrainValidationSplit(estimator=forest,estimatorParamMaps=paramGrid,evaluator=evaluator2,trainRatio=0.8)
    (training, test) = pca_features_rating.randomSplit([0.8, 0.2])

    model = tvs.fit(training)
    model.save(model_path)
    best = model.bestModel
    prediction = model.transform(test)
    acc = evaluator.evaluate(prediction)
    roc = evaluator2.evaluate(prediction)
    
    return prediction, acc, roc, best


def remove_outliers(df):
    for column in df.columns:
        if not column in ['BGGId', 'Description', 'Name', 'GoodPlayers', 'Family', 'ImagePath', 'Themes',
                        'Mechanics', 'Categories', 'Subcategories', 'Artists', 'Designers', 'Publishers',
                        'NumComments', 'BestPlayers', 'Kickstarted', 'IsReimplementation']:
            print(column)

    for column in df.columns:
        if not column in ['BGGId', 'Description', 'Name', 'GoodPlayers', 'Family', 'ImagePath', 'Themes',
                        'Mechanics', 'Categories', 'Subcategories', 'Artists', 'Designers', 'Publishers',
                        'NumComments', 'BestPlayers', 'Kickstarted', 'IsReimplementation']:
            # Outliers removal
            if column != 'YearPublished':
                quant_df = df.quantile([0.25, 0.75])
                IQR = stats.iqr(df[column])
                if IQR != 0:
                    df = df[(df[column] > quant_df.loc[0.25][column] - IQR) & (df[column] < quant_df.loc[0.75][column] + IQR)]
                else:
                    df = df[(df[column] >= quant_df.loc[0.25][column]) & (df[column] <= quant_df.loc[0.75][column])]
            else:
                df = df[df[column] > 1970]

            print('{} {}'.format(column, df.shape[0]))

    return df
