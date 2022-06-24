import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from tqdm import tqdm
from utils import *

color_list = plt.cm.Pastel1(np.arange(9))


if __name__ == '__main__':

    y_pred_array = []
    at = np.array([0.1, 1, 10])
    ATs = np.array(list(itertools.product(at, at)))
    BTs = np.ones((3*3, 2))

    Xs = np.concatenate([ATs, BTs], axis=1).astype(float)

    Ys_list = list(itertools.product([0, 1], repeat=9))
    pred_Ys_list = np.load('y_pred_array.npy')

    loss_list = []

    for i in range(len(Ys_list)):

        Ys = Ys_list[i]
        pred_Ys = pred_Ys_list[i]
        Ys_str = ''.join([str(i) for i in Ys])

        loss = mse(Ys, pred_Ys)
        loss_list.append(loss)


        pred_Ys = np.array(pred_Ys).reshape(3, 3)


    loss_list = np.array(loss_list)

    print(np.sum(loss_list>0.05))
    print(np.sum(loss_list>0.1))
    print(np.sum(loss_list>0.15))
    print(np.quantile(loss_list, q=np.arange(0, 1, 0.1)))
        
    print(loss_list)
    plt.hist(loss_list, bins=np.arange(0, 0.3, 0.01), color=color_list[0])
    plt.xlabel('MSE')
    plt.ylabel('counts')
    plt.title('MSE_dist')
    plt.savefig('MSE_dist.png')
    plt.show()
