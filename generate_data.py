import os
import numpy as np
import itertools
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm

import models
from utils import *
from plot_utils import *

from copy import deepcopy




index = np.array(list(itertools.product(np.linspace(0, 1, 2), repeat=2)))
X = np.power(10, 2*index-1)

truth_table_list = np.array(list(itertools.product([0, 1], repeat=4))).reshape(16, 2, 2)
index_digital = (index > 0.5).astype(int)
Y_list = truth_table_list[:, index_digital[:,0], index_digital[:,1]].astype(float)

np.save('data/2_input_boolean_x.npy', X)
np.save('data/2_input_boolean_y.npy', Y_list)
print(X.shape)
print(Y_list.shape)
plot_patterns_imshow(Y_list.reshape(4,4,2,2))




index = np.array(list(itertools.product(np.linspace(0, 1, 2), repeat=3)))
X = np.power(10.0, 2*index-1)

truth_table_list = np.array(list(itertools.product([0, 1], repeat=8))).reshape(256, 2, 2, 2)
index = index.astype(int)
Y_list = truth_table_list[:, index[:,0], index[:,1], index[:,2]].astype(float)

np.save('data/3_input_boolean_x.npy', X)
np.save('data/3_input_boolean_y.npy', Y_list)
print(X.shape)
print(Y_list.shape)
plot_patterns_imshow(Y_list.reshape(16,16,4,2))




index = np.array(list(itertools.product(np.linspace(0, 1, 3), repeat=2))).astype(int)
X = np.power(10.0, index-1)

truth_table_list = np.array(list(itertools.product([0, 1], repeat=9))).reshape(512, 3, 3)
Y_list = truth_table_list[:, index[:,0], index[:,1]].astype(float)

np.save('data/2_input_3_quantized_boolean_x.npy', X)
np.save('data/2_input_3_quantized_boolean_y.npy', Y_list)
print(X.shape)
print(Y_list.shape)
plot_patterns_imshow(Y_list.reshape(16,32,3,3))




index = np.array(list(itertools.product(np.linspace(0, 1, 10), repeat=2)))
X = np.power(10, 2*index-1)

truth_table_list = np.array(list(itertools.product([0, 1], repeat=4))).reshape(16, 2, 2)
index_digital = (index > 0.5).astype(int)
Y_list = truth_table_list[:, index_digital[:,0], index_digital[:,1]].astype(float)

np.save('data/2_input_boolean_10_x.npy', X)
np.save('data/2_input_boolean_10_y.npy', Y_list)
print(X.shape)
print(Y_list.shape)
plot_patterns_imshow(Y_list.reshape(4,4,10,10))




index = np.array(list(itertools.product(np.linspace(0, 1, 10), repeat=2)))
X = np.power(10, 2*index-1)

truth_table_list = np.array(list(itertools.product([0, 1], repeat=4))).reshape(16, 2, 2)
Y_list = np.array([bilinear_interpolate(Y, index[:, 0], index[:, 1]) for Y in truth_table_list]).astype(float)

np.save('data/2_input_analog_10_x.npy', X)
np.save('data/2_input_analog_10_y.npy', Y_list)
print(X.shape)
print(Y_list.shape)
plot_patterns_imshow(Y_list.reshape(4,4,10,10))




index = np.array(list(itertools.product(np.linspace(0, 1, 10), repeat=3)))
X = np.power(10.0, 2*index-1)

truth_table_list = np.array(list(itertools.product([0, 1], repeat=8))).reshape(256, 2, 2, 2)
index = index.astype(int)
Y_list = truth_table_list[:, index[:,0], index[:,1], index[:,2]].astype(float)

np.save('data/3_input_boolean_10_x.npy', X)
np.save('data/3_input_boolean_10_y.npy', Y_list)
print(X.shape)
print(Y_list.shape)
# plot_patterns_imshow(Y_list.reshape(16,16,4,2))