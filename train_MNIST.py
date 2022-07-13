import os
import sys
import time
import argparse
import itertools
import logging
import logging.config

import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, datasets, transforms
from tqdm import tqdm

from models.networks import CompetitiveNetwork
from models.mlp import MLP
from utils.utils import *

config = parse_config('config_mnist.yaml')

set_seed(config['seed'])

logging.config.dictConfig(config['logger'])
logger = logging.getLogger()

nA = config['nA']
nB = config['nB']
nY = config['nY']
max_epochs = config['max_epochs']
log_epochs = config['log_epochs']


# train
def train(model, device, criterion, optimizer, dataloader, max_epochs, log_epochs, train=True):

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
                # 模型只能输入一个样本
                y_pred[j] = model(x[j, :2], x[j, 2:])[0]

            loss = criterion(y, y_pred)
            loss_epoch += loss.item()
            Ys_pred.append(y_pred.detach().numpy())

            if (train == True):
                # 每个batch反向传播一次
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_epochs.append(loss_epoch)

        if (epoch % log_epochs == 0):
            logger.info(f'epoch = {epoch}, loss = {loss_epochs[-1]:.8f}')

        if (epoch > 2*log_epochs):
            if (np.mean(loss_epochs[-2*log_epochs:-log_epochs]) - np.mean(loss_epochs[-log_epochs:]) < 1e-3):
                continue
                break

    Ys_pred = np.array(Ys_pred).reshape(-1)

    return loss_epochs, Ys_pred



def main(config=None):
    train_set = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=16, shuffle=False)

    loss_3 = []
    Ys_pred_3 = []
    for j in range(1):
        # 避免初始化的影响，重复训练3次
        model = CompetitiveNetwork(nA, nB, nY, reparameterize='exp', gradient='linear_algebra')
        # model = MLP(nA, nB, nY)
        device = torch.device("cpu")
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        loss_epochs, Ys_pred = train(model, device, criterion, optimizer, train_loader, max_epochs, log_epochs)
        
        loss_3.append(loss_epochs[-1])
        Ys_pred_3.append(Ys_pred)

        if (loss_epochs[-1] < 1e-2):
            break

    # 选择最佳的模型
    Ys_pred = Ys_pred_3[np.argmin(loss_3)]

    #plt.figure(figsize=(8,6), dpi=100)
    #plt.plot(loss_epochs)
    #plt.show()
    #plt.close()

    logger.info(f'final loss = {loss_epochs[-1]:.8f}')
    logger.info(f'Ys = {Ys}')
    logger.info(f'Ys_pred = {Ys_pred}')
    
    Ys_pred_list.append(Ys_pred)

    Ys_pred_list = np.array(Ys_pred_list)
    #np.save(f'{nB}_Ys_pred_list.npy', Ys_pred_list)



if __name__ == '__main__':
    '''args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default='config.yaml', type=str,
                      help='config file path (default: config.yaml)')
    args = args.parse_args()
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)'''

    main()
