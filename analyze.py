import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from tqdm import tqdm
from utils import *

color_list = plt.cm.Pastel1(np.arange(9))
set_seed(42)
figures_dir = 'figures/figures2v2/'
nA = 2
nB = 2
nY = 1


if __name__ == '__main__':

    at = np.array([0.1, 1, 10])
    ATs = np.array(list(itertools.product(at, at)))
    BTs = np.ones((3*3, nB))

    Ys_list = list(itertools.product([0, 1], repeat=9))
    pred_Ys_list = np.load(str(nB)+'Ys_pred_list.npy')

    loss_list = []

    for i in range(len(Ys_list)):
        Ys = Ys_list[i]
        pred_Ys = pred_Ys_list[i]
        Ys_str = ''.join([str(i) for i in Ys])

        loss = mse(Ys, pred_Ys)
        loss_list.append(loss)

    loss_list = np.array(loss_list)

    print(np.sum(loss_list<0.01) / 512)
    print(np.sum(loss_list<0.02) / 512)
    print(np.sum(loss_list<0.05) / 512)
    print(np.sum(loss_list<0.1) / 512)
    
    # print(np.quantile(loss_list, q=np.arange(0, 1, 0.1)))
    
    
    plt.hist(loss_list, bins=np.arange(0, 0.3, 0.01), color=color_list[0])
    plt.xlabel('MSE')
    plt.ylabel('counts')
    plt.title('MSE_dist')
    plt.savefig('MSE_dist.png')
    plt.show()
