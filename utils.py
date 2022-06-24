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


if __name__ == '__main__':

    at = np.array([0.1, 1, 10])
    ATs = np.array(list(itertools.product(at, at)))

    y_pred_mat = np.array([1,0,0, 0,0.8,0, 0,0,0.6]).reshape(3, 3)
    print('output', y_pred_mat)
    plot_heatmap(y=y_pred_mat, x=at)
