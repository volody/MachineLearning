import os
import pandas as pd  # read file
import numpy as np   # processing functions

def updateDataset(data):
   data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')
   data['inc_angle'] = data['inc_angle'].fillna(method='pad')

current_file = os.path.abspath(os.path.dirname(__file__)) 
input_folder = os.path.join(current_file, '..\..\..\statoil-iceberg-classifier-challenge\input')
train_filename = os.path.join(input_folder, 'train.json')
test_filename = os.path.join(input_folder, 'test.json')

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

train = pd.read_json(train_filename)
updateDataset(train)

test = pd.read_json(test_filename)
updateDataset(test)

print(list(train)) # display columns in data frame

# todo: implement logistic regression for gradient descent
#  y^ = sigmoid ( W^T * x + b)

#
#  sigmoid (z) = 1 / (1 + exp(-z))
#
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_parameters_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

def forward_propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = - (1/m) * np.sum( Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return A, cost

def backward_propagate(w, A, X, Y):
    m = X.shape[1]    
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    return dw, db

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        A, cost = forward_propagate(w, b, X, Y)
        dw, db = backward_propagate(w, A, X, Y)
        w -= learning_rate * dw
        b -= learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return w, b, dw, db, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    assert(Y_prediction.shape == (1, m))
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    w, b, dw, db, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "train_accuracy" : train_accuracy,
         "test_accuracy" : test_accuracy,
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d

train_set_x = train['band_1']
train_set_y = train['is_iceberg']
test_set_x = test['band_1']
test_set_y = test['is_iceberg']

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

print("train accuracy: {} %".format(d["train_accuracy"]))
print("test accuracy: {} %".format(d["test_accuracy"]))

# version 1: implement loss function
# L( y^, y) = (1/2) * (y^ - y)^2
# this loss function creates non optimized problem when
# no local optima for gradient descent

# version 2: implement loss function
# L( y^, y) = - ( y * log(y^) + (1-y) * log(1- y^))

# todo: using loss function implement cost function
# J(w,b) = (1/m) * sum {i= 1, m} ( L(y^[i],y[i]))

# J = 0
# dw1 = 0
# dw2 = 0
# db = 0
# X = train['band_1']
# Y = train['is_iceberg']

# m = X.shape[0]
# print('X.shape[0] = ')
# print(m)

# for i in range(m):
#    z[i] = W.T * X[i] + b
#    a[i] = sigma(z[i])
#    J += -( Y[i] * log(a[i]) + ( 1 - Y[i]) * log(1 - a[i]))
#    dz[i] = a[i] - Y[i]
#    dw += X[i] * dz[i]
#    db += dz[i]

# J=j/m
# dw=dw/m
# db=db/m

print("done")