import os
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_heatmap(y, x=None, fig_name='temp'):
    # 热图
    plt.figure(figsize=(5, 5), dpi=100)
    plt.imshow(y, cmap='Reds', origin='lower', vmin=-0.1, vmax=1.1)

    m, n = y.shape
    if (x is None):
        plt.xticks(np.arange(m), np.arange(m), fontsize=10)
        plt.yticks(np.arange(n), np.arange(m), fontsize=10)
    else:
        plt.xticks(np.arange(m), x, fontsize=10)
        plt.yticks(np.arange(n), x, fontsize=10)

    # 每个方格上标记数值
    for i in range(m):
        for j in range(n):
            text = plt.text(j, i, '{:.3f}'.format(y[i, j]),
                            ha="center", va="center", color="black", fontsize=10)

    plt.title(fig_name)
    #plt.colorbar()
    plt.savefig('figures/' + fig_name + '.png')
    # plt.show()
    plt.close()


if __name__ == '__main__':

    at = np.array([0.1, 1, 10])
    ATs = np.array(list(itertools.product(at, at)))

    y_pred_mat = np.array([1,0,0, 0,0.8,0, 0,0,0.6]).reshape(3, 3)
    print('output', y_pred_mat)
    plot_heatmap(y=y_pred_mat, x=at)
