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
from models.networks import *


set_seed(42)
nA = 2
nB = 3
nY = 1

# train by dataloader
def train(model, device, criterion, optimizer, dataloader, max_epochs, log_steps, train=True):

    model.to(device)
    model.train()
    loss_epochs = []

    for epoch in range(max_epochs+1):
        loss_epoch = 0
        Ys_pred = []

        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            y_pred = torch.Tensor(y.shape).to(device)
            for j in range(len(x)):
                y_pred[j] = model(x[j, :2], x[j, 2:])[0]

            loss = criterion(y, y_pred)
            loss_epoch += loss
            Ys_pred.append(y_pred.detach().numpy())

            if (train == True):
                # 每个batch反向传播一次
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_epochs.append(loss_epoch.item())

        if (epoch % log_steps == 0):
            print(f'epoch = {epoch}, loss = {loss_epochs[-1]:.6f}')

        if (epoch > 2*log_steps):
            if (np.mean(loss_epochs[-2*log_steps:-log_steps]) - np.mean(loss_epochs[-log_steps:]) < 1e-3):
                continue
                break

    Ys_pred = np.array(Ys_pred).reshape(-1)

    return loss_epochs, Ys_pred



def main(args=None):

    Xs, Ys_list = generate_patterns(nA=nA, nB=nB, AT_optionals=np.array([0.1, 1, 10]))
    Ys_list = np.array([[1,0,0, 0,1,0, 0,0,1],
                        [0,0,1, 0,1,0, 1,0,0]])
    Ys_pred_list = []

    for i in tqdm(range(len(Ys_list))):
        print('')
        Ys = Ys_list[i]

        Xs = torch.Tensor(Xs)
        Ys = torch.Tensor(Ys)
        train_ds = TensorDataset(Xs, Ys)
        train_dl = DataLoader(train_ds, batch_size=9, shuffle=False)

        loss_3 = []
        Ys_pred_3 = []
        for j in range(3):
            # 避免初始化的影响，重复训练3次
            model = CompetitiveNetwork(nA, nB, nY, reparameterize='square', gradient='linear_algebra')
            device = torch.device("cpu")
            criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
            max_epochs = 2000
            log_steps = 100

            loss_epochs, Ys_pred = train(model, device, criterion, optimizer, train_dl, max_epochs, log_steps)
            
            loss_3.append(loss_epochs[-1])
            Ys_pred_3.append(Ys_pred)

            if (loss_epochs[-1] < 1e-2):
                break

        # 选择最佳的模型
        Ys_pred = Ys_pred_3[np.argmin(loss_3)]

        plt.figure(figsize=(8,6), dpi=100)
        plt.plot(loss_epochs)
        #plt.savefig(figures_dir + 'loss_epochs.png')
        plt.show()
        plt.close()
        
        #print('K', model.comp_layer.param.data)
        #print('W', model.linear.weight.data)
        #print('B', model.linear.bias.data)

        print(f'final loss = {loss_epochs[-1]:.6f}')
        print('Ys', Ys)
        print('Ys_pred', Ys_pred)
        
        Ys_pred_list.append(Ys_pred)

    Ys_pred_list = np.array(Ys_pred_list)
    #np.save(f'{nB}_Ys_pred_list.npy', Ys_pred_list)



if __name__ == '__main__':
    '''args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--nA', default=2, type=int)
    args.add_argument('--nB', default=2, type=int)
    args.add_argument('--nY', default=1, type=int)
    args = args.parse_args()'''

    main()
