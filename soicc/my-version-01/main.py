import pandas as pd  # read file
import numpy as np   # processing functions

def updateDataset(data):
   data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')
   data['inc_angle'] = data['inc_angle'].fillna(method='pad')

train = pd.read_json('../../../statoil-iceberg-classifier-challenge/input/train.json')
updateDataset(train)

# Data fields
# train.json, test.json
# 
# The data (train.json, test.json) is presented in json format. The files consist of a list of images, 
# and for each image, you can find the following fields:
# 
#     id - the id of the image
#     band_1, band_2 - the flattened image data. Each band has 75x75 pixel values in the list
#     inc_angle - the incidence angle of which the image was taken. 
#     is_iceberg - the target variable, set to 1 if it is an iceberg, and 0 if it is a ship.

print(list(train)) # display columns in data frame

# todo: implement logistic regression for gradient descent
#  y^ = sigmoid ( W^T * x + b)

#
#  sigmoid (z) = 1 / (1 + exp(-z))
#
def sigmoid(z):
    return  1 / (1 + exp(-z))

# version 1: implement loss function
# L( y^, y) = (1/2) * (y^ - y)^2
# this loss function creates non optimized problem when
# no local optima for gradient descent

# version 2: implement loss function
# L( y^, y) = - ( y * log(y^) + (1-y) * log(1- y^))

# todo: using loss function implement cost function
# J(w,b) = (1/m) * sum {i= 1, m} ( L(y^[i],y[i]))

J = 0
dw1 = 0
dw2 = 0
db = 0
X = train['band_1']
Y = train['is_iceberg']

m = X.shape[0]
print('X.shape[0] = ')
print(m)

for i in range(m):
   z[i] = W.T * X[i] + b
   a[i] = sigma(z[i])
   J += -( Y[i] * log(a[i]) + ( 1 - Y[i]) * log(1 - a[i]))
   dz[i] = a[i] - Y[i]
   dw += X[i] * dz[i]
   db += dz[i]

J=j/m
dw=dw/m
db=db/m

print("done")