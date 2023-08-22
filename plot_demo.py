import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm

import models
from utils import *

from copy import deepcopy





def get_file_paths(folder_path):
    file_name_list = sorted([name for name in os.listdir(folder_path) if name[-3:]=='npy'])
    file_path_list = [os.path.join(folder_path, file_name) for file_name in file_name_list]
    return file_path_list


def get_best_result_list(file_path, pattern_num=None):
    result_list = np.load(file_path, allow_pickle=True)#.reshape(pattern_num, -1)
    best_result_list = []
    for i in range(len(result_list)):
        #best_result = sorted(list(result_list[i]), key=lambda result: result['valid_loss'])[0]
        best_result = sorted(list(result_list[i]), key=lambda result: result['best_valid_loss'])[0]
        best_result_list.append(best_result)
    return best_result_list


def get_Y_pred(model, X):
    test_dataset = TensorDataset(torch.FloatTensor(X))
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False, drop_last=False)
    Y_pred = []
    for (x,) in test_dataloader:
        out = model(x)
        Y_pred.extend(out.detach().numpy())
    Y_pred = np.array(Y_pred)
    return Y_pred


def get_Y_pred_list(result_list, X):
    Y_pred_list = []
    for i in range(len(result_list)):
        model = init_obj(models, result_list[i]['config']['model'])
        if type(result_list[i]['best_model_state_dict']) is bytes:
            model.load_state_dict(pickle.loads(result_list[i]['best_model_state_dict']))
        else:
            model.load_state_dict(result_list[i]['best_model_state_dict'])
        Y_pred = get_Y_pred(model, X)
        Y_pred_list.append(Y_pred)
    Y_pred_list = np.array(Y_pred_list)
    return Y_pred_list


def get_Y_pred_array(folder_path, pattern_num, x_filename):
    file_name_list = sorted([name for name in os.listdir(folder_path) if name.find('npy') != -1])
    file_path_list = [folder_path+'/'+file_name for file_name in file_name_list]
    best_result_array = [get_best_result_list(file_path, pattern_num) for file_path in tqdm(file_path_list)]

    X = np.load(x_filename)
    Y_pred_array = np.array([get_Y_pred_list(best_result_list, X) for best_result_list in tqdm(best_result_array)])
    print(Y_pred_array.shape)
    return Y_pred_array


if __name__ == '__main__':
    Y_pred_array = get_Y_pred_array('results/2_input_boolean/20230822_151024', 16, 'data/2_input_boolean_x.npy')
    np.save('Y_pred_array_2.npy', Y_pred_array)

    Y_pred_array = get_Y_pred_array('results/2_input_boolean/20230822_151024', 16, 'data/2_input_boolean_10_x.npy')
    np.save('Y_pred_array_2_10.npy', Y_pred_array)

    X_2 = np.load('data/2_input_boolean_x.npy')
    Y_list_2 = np.load('data/2_input_boolean_y.npy')
    
    Y_pred_array_2 = np.load('Y_pred_array_2.npy')
    Y_pred_array_2_10 = np.load('Y_pred_array_2_10.npy')

    plot_patterns_imshow(Y_pred_array_2_10[0].reshape(4, 4, 10, 10), dpi=100, file_name='0.svg')
