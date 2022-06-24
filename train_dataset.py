import os
import sys
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import argparse

from utils import *
from models.competitive_network import *


np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)

set_seed(42)
figures_dir = 'figures22/'


def train(model, device, criterion, optimizer, dataloader, dataset, max_epochs, log_steps, train=True):

    model.to(device)
    model.train()
    losses = []

    for epoch in range(max_epochs+1):
        loss_sum = 0
        Ys_pred = []
        # 手动mini batch
        for i, (x, y) in enumerate(dataset):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x[:2], x[2:])

            loss = criterion(y, y_pred[0])
            loss_sum += loss
            Ys_pred.append(y_pred.detach().numpy())

        if (train == True):
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

        losses.append(loss_sum.item())
        # print(loss_sum.item())

        if (epoch % log_steps == 0):
            print(f'epoch = {epoch}, loss = {loss_sum.item():.6f}')

        if (epoch > 2*log_steps):
            if (np.mean(losses[-2*log_steps:-log_steps]) - np.mean(losses[-log_steps:]) < 1e-3):
                break

    print('K', model.comp_layer.param.data)
    print('W', model.linear.weight.data)
    print('B', model.linear.bias.data)

    plt.figure(figsize=(8,6), dpi=100)
    plt.plot(losses)
    plt.savefig(figures_dir + 'loss_list.png')
    #plt.show()
    plt.close()

    return losses[-1], Ys_pred



def main(args):
    
    nA = 2
    nB = 3
    nY = 1
    nB = args.nB

    AT = np.array([0.1, 1, 10])
    ATs = np.array(list(itertools.product(AT, AT)))
    BTs = np.ones((3*3, nB))
    Xs = np.concatenate([ATs, BTs], axis=1)
    Ys_list = list(itertools.product([0, 1], repeat=9))
    '''Ys_list = np.array([[1,0,0, 0,1,0, 0,0,1],
                        [0,0,1, 0,1,0, 1,0,0],
                        [0,1,0, 1,0,1, 1,0,1],
                        [1,0,1, 1,0,1, 0,1,0]])'''
    Ys_pred_list = []

    for i in tqdm(range(3)):
        Ys = Ys_list[2]
        Ys_str = ''.join([str(i) for i in Ys])
        print('')

        Xs = torch.Tensor(Xs)
        Ys = torch.Tensor(Ys)
        train_ds = TensorDataset(Xs, Ys)
        # train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)

        for j in range(3):
            # 避免初始化的影响，重复训练3次
            model = CompetitiveNetwork(nA, nB, nY, reparameterize='square', gradient='linear_algebra')
            device = torch.device("cpu")
            criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
            # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) #参数稀疏性不好
            max_epochs = 2000
            log_steps = 100

            loss, Ys_pred = train(model, device, criterion, optimizer, None, train_ds, max_epochs, log_steps)
            if (loss < 0.1):
                break

        Ys_pred = np.array(Ys_pred).reshape(-1)
        print('Ys', Ys)
        print('Ys_pred', Ys_pred)
        plot_heatmap(y=Ys_pred.reshape(3, 3), x=AT, title=Ys_str+f'_{loss:.3f}')

        Ys_pred_list.append(Ys_pred)

    Ys_pred_list = np.array(Ys_pred_list)
    np.save(str(nB)+'Ys_pred_list.npy', Ys_pred_list)



def plot_heatmap(y, x=None, title='temp'):
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
    plt.savefig(figures_dir + title + '.png')
    #plt.show()
    plt.close()



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--nA', default=2, type=int)
    args.add_argument('--nB', default=2, type=int)
    args.add_argument('--nY', default=1, type=int)

    args = args.parse_args()

    main(args)
