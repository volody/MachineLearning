import os
import pandas as pd  # read file
import numpy as np   # processing functions


def updateDataset(data):
   data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')
   data['inc_angle'] = data['inc_angle'].fillna(method='pad')


def normalize(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x

# todo: implement code here

print("start")

current_file = os.path.abspath(os.path.dirname(__file__))
input_folder = os.path.join(
    current_file, '..\..\..\statoil-iceberg-classifier-challenge\input')
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
m_train = train.shape[0]

test = pd.read_json(test_filename)
updateDataset(test)

to_arr = lambda x: np.asarray([np.asarray(item) for item in x])

split_value = 0.92
train_number = int(m_train * split_value)

train_set_x, dev_set_x = np.split(
    to_arr(train['band_1'].values), [train_number])
train_set_y, dev_set_y = np.split(
    train['is_iceberg'].values, [train_number])

print("train_set_y iceberg count is {}".format(np.count_nonzero(train_set_y)))
print("dev_set_y iceberg count is {}".format(np.count_nonzero(dev_set_y)))

train_set_x = normalize(train_set_x.T)
train_set_y = normalize(train_set_y.reshape(1, train_number))

dev_set_x = normalize(dev_set_x.T)
dev_set_y = normalize(dev_set_y.reshape(1, m_train - train_number))

print("done")
