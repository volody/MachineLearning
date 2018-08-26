# Load libraries for analysis and visualization
import pandas as pd 
import numpy as np
from sklearn import linear_model

# Load in the train and test datasets from the CSV files
X = pd.read_csv('../../input/titanic/train.csv')
#test = pd.read_csv('../../input/titanic/test.csv')

# adjust data
y = X['Survived']
X.drop(['Survived'], axis=1)

# Binary classifier
X_train, X_test, y_train, y_test = X[:800], X[800:], y[:800], y[800:]

y_train_survived = (y_train == 1)
y_test_survived = (y_test == 1)

sgd_clf = linear_model.SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_survived)

sgd_clf.predict([500])