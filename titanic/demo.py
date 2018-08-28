# Load libraries for analysis and visualization
import pandas as pd 
import numpy as np
from sklearn import linear_model
import random

def fillNaN_with_unifrand(df, name):
    col = df[name]
    a = col.values
    m = np.isnan(a) # mask of NaNs
    mu, sigma = col.mean(), col.std()
    a[m] = np.random.normal(mu, sigma, size=m.sum())
    return df

# Load in the train and test datasets from the CSV files
# into DataFrame
X = pd.read_csv('../../input/titanic/train.csv')
#test = pd.read_csv('../../input/titanic/test.csv')

# adjust data
y = X['Survived']
# convert string to enum
X['Sex1'] = pd.factorize(X['Sex'])[0] + 1
X['Embarked1'] = pd.factorize(X['Embarked'])[0] + 1
# drop string column
X = X.drop(columns=['Survived','Name','Ticket','Sex','Embarked','PassengerId','Cabin'])
fillNaN_with_unifrand(X,'Age')
#x['A'] = x['A'].apply(lambda v: random.random() * 1000)

print(X.head(20))

# Binary classifier
X_train, X_test, y_train, y_test = X[:800], X[800:], y[:800], y[800:]

y_train_survived = (y_train == 1)
y_test_survived = (y_test == 1)

sgd_clf = linear_model.SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_survived)

some_data = X_test.iloc[[10]]
sgd_clf.predict([some_data])