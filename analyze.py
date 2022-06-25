import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from tqdm import tqdm
from utils import *


set_seed(42)

nA = 2
nB = 2
nY = 1

figures_dir = f'figures/2v{nB}/'
print(figures_dir)
color_list = plt.cm.Pastel1(np.arange(9))

def plot_heatmap(y, x=None, title='temp', show=False, save=False):
    # 热图
    m, n = y.shape

    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(y, cmap='Reds', origin='lower', vmin=-0.1, vmax=1.1)
    if (x is None):
        plt.xticks(np.arange(m), np.arange(m), fontsize=10)
        plt.yticks(np.arange(n), np.arange(m), fontsize=10)
    else:
        plt.xticks(np.arange(m), x, fontsize=10)
        plt.yticks(np.arange(n), x, fontsize=10)

    # 每个方格上标记数值
    for i in range(m):
        for j in range(n):
            plt.text(j, i, '{:.3f}'.format(y[i, j]), 
                     ha="center", va="center", color="black", fontsize=10)

    plt.title(title)

    if (save==True):
        plt.savefig(figures_dir + title + '.png')
    if (show==True):
        plt.show()
    plt.close()



def main():

    AT = np.array([0.1, 1, 10])
    ATs = np.array(list(itertools.product(AT, AT)))
    BTs = np.ones((3*3, nB))

    Ys_list = list(itertools.product([0, 1], repeat=9))
    Ys_pred_list = np.load(f'{nB}Ys_pred_list.npy')

    loss_list = []

    for i in tqdm(range(len(Ys_list))):
        Ys = Ys_list[i]
        Ys_str = ''.join([str(i) for i in Ys])
        Ys_pred = Ys_pred_list[i]

        loss = mse(Ys, Ys_pred)
        # plot_heatmap(y=Ys_pred.reshape(3, 3), x=AT, title=Ys_str+f'_{loss:.3f}', show=False, save=True)

        loss_list.append(loss)

    loss_list = np.array(loss_list)

    count_list = []
    threshold_list = [0.005, 0.01, 0.02, 0.05, 0.1]
    for threshold in threshold_list:
        count_list.append(np.sum(loss_list<threshold))
    count_list = np.array(count_list)
    
    print('threshold', threshold_list)
    print('ratio', count_list / 512)
    
    plt.hist(loss_list, bins=np.arange(0, 0.3, 0.01), color=color_list[0])
    plt.xlabel('MSE')
    plt.ylabel('counts')
    plt.title('MSE_dist')
    #plt.savefig('MSE_dist.png')
    #plt.show()
    plt.close()

    data.append(count_list)




if __name__ == '__main__':
    data = []
    for nB in [2,3,4,5]:
        main()

    data = np.array(data)
