import os
import numpy as np
import itertools
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pickle

from utils import *


color_list = np.delete(plt.cm.get_cmap('Set3')(np.arange(12)), 1, axis=0)

# params={'font.serif':'Times New Roman', # 'Arial',
#         'font.style':'normal', # 'italic'
#         'font.weight':'normal', # 'blod'
#         'font.size':12,
#         }
# plt.rcParams.update(params)


font_kwargs = {'family': 'sans-serif',
               'sans-serif': 'Arial',
               'size': 10}
mathtext_kwargs = {'fontset': 'custom',
                   'bf': 'Arial:bold',
                   'cal': 'Arial:italic',
                   'it': 'Arial:italic',
                   'rm': 'Arial'}
savefig_kwargs = {'dpi': 400,
                  'bbox_inches': 'tight',
                  'transparent': True}
plt.rc('font', **font_kwargs)
plt.rc('mathtext', **mathtext_kwargs)




def plot_patterns_imshow(data, file_name='temp.svg', dpi=100):
    if len(data.shape) == 2:
        data = data[None, None, :, :]
    elif len(data.shape) == 3:
        data = data[None, :, :, :]
    elif len(data.shape) == 4:
        pass
    
    nrows, ncols, length, width = data.shape
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows), dpi=dpi)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    for i in range(nrows):
        for j in range(ncols):
            img = axes[i, j].imshow(data[i, j], origin='lower', vmin=0, vmax=1, cmap='OrRd')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_aspect('equal')
            # axes[i, j].spines["top"].set_visible(False)
            # axes[i, j].spines["bottom"].set_visible(False)
            # axes[i, j].spines["left"].set_visible(False)
            # axes[i, j].spines["right"].set_visible(False)
            
    plt.savefig(file_name, **savefig_kwargs)
    plt.show()
    plt.close()


def plot_patterns_pcolormesh(data, file_name='temp.svg', dpi=200):
    if len(data.shape) == 2:
        data = data[None, None, :, :]
    elif len(data.shape) == 3:
        data = data[None, :, :, :]
    elif len(data.shape) == 4:
        pass
    
    nrows, ncols, length, width = data.shape
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows), dpi=dpi)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    for i in range(nrows):
        for j in range(ncols):
            img = axes[i, j].pcolormesh(data[i, j], vmin=0, vmax=1, cmap='OrRd', edgecolors='black', linewidth=0.1)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_aspect('equal')
            # axes[i, j].spines["top"].set_visible(False)
            # axes[i, j].spines["bottom"].set_visible(False)
            # axes[i, j].spines["left"].set_visible(False)
            # axes[i, j].spines["right"].set_visible(False)
            
    plt.savefig(file_name, **savefig_kwargs)
    plt.show()
    plt.close()

# def plot_patterns_scatter(X, Y_list):
#     if (len(Y_list.shape)) == 3:
#         nrows, ncols, _ = Y_list.shape
#         fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols/2, nrows/2), dpi=100)
#         axes = axes.reshape(nrows, ncols)
#         for i in range(nrows):
#             for j in range(ncols):
#                 img = axes[i, j].scatter(x=X[:, 0], y=X[:, 1], c=Y_list[i, j], s=1, vmin=0, vmax=1)
#                 axes[i, j].set_xticks([])
#                 axes[i, j].set_yticks([])
#         plt.show()
#         plt.close()


# def plot_lines(data, x_ticks=None, y_ticks=None, labels=None, x_label=None, y_label=None, title=None):
#     if x_ticks is None:
#         x_ticks = np.arange(1, len(data[0])+1)
#     if y_ticks is None:
#         pass
#     if labels is None:
#         labels = np.arange(1, len(data)+1)
#     if x_label is None:
#         x_label = 'nB'
#     if y_label is None:
#         y_label = 'MSE'
#     if title is None:
#         title = ''
        
#     plt.figure(figsize=(8, 6), dpi=100)
#     for i in range(len(data)):
#         p = plt.plot(x_ticks, data[i], 'o-', color=color_list[i], label=labels[i])
#     plt.legend(fontsize=10, loc=1)
#     if x_ticks is not None:
#         plt.xticks(x_ticks)
#     if y_ticks is not None:
#         plt.yticks(y_ticks)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(title)
#     plt.show()


def digital_operations(truth_table_list, X):
    X_d = (X > 0.5).astype(int)
    if X.shape[1] == 2:
        Y_list = np.array([Y[X_d[:, 0], X_d[:, 1]] for Y in truth_table_list])
    if X.shape[1] == 3:
        Y_list = np.array([Y[X_d[:, 0], X_d[:, 1], X_d[:, 2]] for Y in truth_table_list])
    return Y_list


def bilinear_interpolate(img, x, y):
    xrange, yrange = img.shape

    x0 = np.minimum(np.floor(x).astype(int), xrange-2)
    x1 = x0 + 1
    y0 = np.minimum(np.floor(y).astype(int), yrange-2)
    y1 = y0 + 1

    Ia = img[x0, y0]
    Ib = img[x0, y1]
    Ic = img[x1, y0]
    Id = img[x1, y1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    ans = wa*Ia + wb*Ib + wc*Ic + wd*Id
    return ans


def analog_operations(truth_table_list, X):
    Y_list = np.array([bilinear_interpolate(Y, X[:, 0], X[:, 1]) for Y in truth_table_list])
    return Y_list



def BCELoss(y_pred, y_true):
    loss = -(y_true * np.log(y_pred+1e-9) + (1-y_true) * (np.log(1-y_pred+1e-9)))
    return loss.mean(-1)

def MSELoss(y_pred, y_true):
    loss = (y_true - y_pred) ** 2
    return loss.mean(-1)


# def generate_2_2(num=2):
#     np.random.seed(0)
#     index = np.array(list(itertools.product(np.linspace(0, 1, num), repeat=2)))
#     X = np.power(10, 2*index-1)

#     truth_table_list = np.array(list(itertools.product([0, 1], repeat=4))).reshape(16, 2, 2)
#     index_digital = (index > 0.5).astype(int)
#     Y_list = truth_table_list[:, index_digital[:,0], index_digital[:,1]]

#     return X, Y_list




# def bin2dec(b):
#     d = 0
#     l = len(b)
#     for i in range(l):
#         d += b[i] * (2 ** (l-i-1))
#     return d


# def get_min_index(Ys, dim=3):
#     indice = []
#     for perm_index in list(itertools.permutations(np.arange(dim))):
#         indice.append(bin2dec(Ys.transpose(perm_index).reshape(-1)))
#     return np.min(indice)


# def get_no_repeat_truth_table_list(truth_table_list):
#     no_repeat_index_list = []
#     no_repeat_truth_table_list = []
#     for i in range(len(truth_table_list)):
#         index = get_min_index(truth_table_list[i])
#         if index not in no_repeat_index_list:
#             no_repeat_index_list.append(index)
#             no_repeat_truth_table_list.append(truth_table_list[i])
#     no_repeat_truth_table_list = np.array(no_repeat_truth_table_list)
#     return no_repeat_truth_table_list


# def generate_2_3():
#     truth_table_list = np.array(list(itertools.product([0, 1], repeat=8))).reshape(256, 2, 2, 2)
#     # truth_table_list = get_no_repeat_truth_table_list(truth_table_list)

#     np.random.seed(0)
#     index = np.array(list(itertools.product([0,1], repeat=3)))
#     X = np.power(10.0, 2*index-1)
#     index = index.astype(int)
#     Y_list = truth_table_list[:, index[:,0], index[:,1], index[:,2]]

#     return X, Y_list



# def generate_3_2():
#     truth_table_list = np.array(list(itertools.product([0, 1], repeat=9))).reshape(512, 3, 3)

#     # no_repeat_list = []
#     # for i in range(len(truth_table_list)):
#     #     repeat_flag = False
#     #     for j in range(len(no_repeat_list)):
#     #         if (truth_table_list[i].T == no_repeat_list[j]).all():
#     #             repeat_flag = True
#     #             break
#     #     if repeat_flag is False:
#     #         no_repeat_list.append(truth_table_list[i])
#     # no_repeat_list = np.array(no_repeat_list)

#     np.random.seed(0)
#     index = np.array(list(itertools.product([0,1,2], repeat=2))).astype(int)
#     X = np.power(10.0, index-1)
#     Y_list = truth_table_list[:, index[:,0], index[:,1]]

#     return X, Y_list



