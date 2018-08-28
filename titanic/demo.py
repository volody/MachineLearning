# 
import pandas as pd 
import numpy as np
from sklearn import base, pipeline, preprocessing, svm, model_selection, ensemble

#import sklearn
#print('The scikit-learn version is {}.'.format(sklearn.__version__))

from future_encoders import OneHotEncoder

# Load in the train and test datasets from the CSV files
# into DataFrame
train_data = pd.read_csv('../../input/titanic/train.csv')
test_data = pd.read_csv('../../input/titanic/test.csv')

# select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

# build the pipeline for the numerical attributes
imputer = preprocessing.Imputer(strategy="median")

num_pipeline = pipeline.Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", preprocessing.Imputer(strategy="median")),
    ])

num_pipeline.fit_transform(train_data)

#We will also need an imputer for the string categorical columns (the regular Imputer does not work on those):
# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(base.BaseEstimator, base.TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

cat_pipeline = pipeline.Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

cat_pipeline.fit_transform(train_data)

preprocess_pipeline = pipeline.FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

X_train = preprocess_pipeline.fit_transform(train_data)
print('X_train')
print(X_train[:5])

y_train = train_data["Survived"]

# SVC
svm_clf = svm.SVC()
svm_clf.fit(X_train, y_train)
print('svm_clf')
print(svm_clf)

# CHECK PREDICION
X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)

# print cross_val_score
svm_scores = model_selection.cross_val_score(svm_clf, X_train, y_train, cv=10)
print('The SVC cross_val_score is {}.'.format(svm_scores.mean()))

# RandomForestClassifier
forest_clf = ensemble.RandomForestClassifier(random_state=42)
forest_scores = model_selection.cross_val_score(forest_clf, X_train, y_train, cv=10)
print('The RandomForestClassifier cross_val_score is {}.'.format(forest_scores.mean()))
