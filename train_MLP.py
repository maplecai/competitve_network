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

from utils.utils import *
from models.mlp import *


set_seed(42)
nA = 2
nB = 2
nY = 1


def train(model, device, criterion, optimizer, dataloader, dataset, max_epochs, log_steps, train=True):

    model.to(device)
    model.train()
    losses = []

    for epoch in range(max_epochs+1):
        loss_batch = 0
        Ys_pred = []
        # 手动 dataloader
        for i, (x, y) in enumerate(dataset):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x[:2], x[2:])
            loss = criterion(y, y_pred[0])
            loss_batch += loss
            Ys_pred.append(y_pred.detach().numpy())

        loss_batch = loss_batch / len(dataset)
        if (train == True):
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        losses.append(loss_batch.item())
        
        if (epoch % log_steps == 0):
            print(f'epoch = {epoch}, loss = {losses[-1]:.6f}')

        if (epoch > 2*log_steps):
            if (np.mean(losses[-2*log_steps:-log_steps]) - np.mean(losses[-log_steps:]) < 1e-4):
                break

    Ys_pred = np.array(Ys_pred).reshape(-1)

    return losses, Ys_pred



def main(args=None):

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

    for i in tqdm(range(len(Ys_list))):
        print('')
        Ys = Ys_list[i]

        Xs = torch.Tensor(Xs)
        Ys = torch.Tensor(Ys)
        train_ds = TensorDataset(Xs, Ys)

        loss_3 = []
        Ys_pred_3 = []
        for j in range(3):
            # 避免初始化的影响，重复训练3次
            model = MLP(nA, nB, nY)
            device = torch.device("cpu")
            criterion = nn.MSELoss()
            #optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) #参数稀疏性不好
            max_epochs = 2000
            log_steps = 100

            losses, Ys_pred = train(model, device, criterion, optimizer, None, train_ds, max_epochs, log_steps)

            loss_3.append(losses[-1])
            Ys_pred_3.append(Ys_pred)

            if (losses[-1] < 1e-2):
                break

        # 选择最佳的模型
        Ys_pred = Ys_pred_3[np.argmin(loss_3)]

        #plt.figure(figsize=(8,6), dpi=100)
        #plt.plot(losses)
        #plt.savefig(figures_dir + 'losses.png')
        #plt.show()
        #plt.close()
        
        #print('K', model.comp_layer.param.data)
        #print('W', model.linear.weight.data)
        #print('B', model.linear.bias.data)

        print(f'final loss = {losses[-1]:.6f}')
        print('Ys', Ys)
        print('Ys_pred', Ys_pred)
        
        Ys_pred_list.append(Ys_pred)

    Ys_pred_list = np.array(Ys_pred_list)
    np.save(f'MLP_{nB}_Ys_pred_list.npy', Ys_pred_list)



if __name__ == '__main__':
    '''args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--nA', default=2, type=int)
    args.add_argument('--nB', default=2, type=int)
    args.add_argument('--nY', default=1, type=int)

    args = args.parse_args()'''

    main()
