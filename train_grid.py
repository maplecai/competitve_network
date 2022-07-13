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
from tqdm import tqdm

from models.networks import CompetitiveNetwork
from models.mlp import MLP
from utils.utils import *


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

        '''if (loss_epoch < 1e-2):
            break'''

        '''if (epoch > 2*log_epochs):
            if (np.mean(loss_epochs[-2*log_epochs:-log_epochs]) - np.mean(loss_epochs[-log_epochs:]) < 1e-3):
                break'''

    Ys_pred = np.array(Ys_pred).reshape(-1)

    return loss_epochs, Ys_pred



def main(config):

    set_seed(config['seed'])
    nA = config['nA']
    nB = config['nB']
    nY = config['nY']
    max_epochs = config['max_epochs']
    log_epochs = config['log_epochs']

    config['logger']['handlers']['file_handler']['filename'] = f'logs/{nA}v{nB}.log'
    logging.config.dictConfig(config['logger'])
    logger = logging.getLogger()


    Xs = generate_Xs(nA=nA, nB=nB, AT_optionals=np.array([0.1, 1, 10]))
    Ys_list = generate_Ys_list(dim=3)
    '''Ys_list = np.array([[1,0,0, 0,1,0, 0,0,1],
                        [0,0,1, 0,1,0, 1,0,0]])'''
    Ys_pred_list = []

    for i in tqdm(range(len(Ys_list))):
        Ys = Ys_list[i]

        Xs = torch.Tensor(Xs)
        Ys = torch.Tensor(Ys)
        train_ds = TensorDataset(Xs, Ys)
        train_dl = DataLoader(train_ds, batch_size=9, shuffle=False, drop_last=True)

        loss_3 = []
        Ys_pred_3 = []
        for j in range(3):
            # 避免初始化的影响，重复训练3次
            model = CompetitiveNetwork(nA, nB, nY, reparameterize='exp')
            device = torch.device("cpu")
            criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

            loss_epochs, Ys_pred = train(model, device, criterion, optimizer, train_dl, max_epochs, log_epochs)
            
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
    np.save(f'2v{nB}_Ys_pred_list.npy', Ys_pred_list)



if __name__ == '__main__':
    '''args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default='config.yaml', type=str,
                      help='config file path (default: config.yaml)')
    args = args.parse_args()
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)'''

    config = parse_config('config.yaml')

    for nB in [2,3,4,5,6]:
        config['nB'] = nB
        main(config)
