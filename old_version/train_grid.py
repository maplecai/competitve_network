import os
import sys
import time
import signal
import datetime
import argparse
import itertools
import multiprocessing
import logging
import logging.config

import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.models import CompetitiveNetwork
from models.mlp import MLP
from utils.utils import *


def train(model, device, criterion, optimizer, dataloader, max_epochs, log_epochs, logger, train=True):

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
            logger.debug(f'epoch = {epoch}, loss = {loss_epochs[-1]:.8f}')

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

    data_dir = config['data_dir']
    figure_dir = config['figure_dir']
    output_dir = config['output_dir']
    log_dir = config['log_dir']

    max_epochs = config['max_epochs']
    log_epochs = config['log_epochs']
    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    config['logger']['handlers']['file_handler']['filename'] = log_dir + f'{nA}v{nB}_{now_time}.log'
    logging.config.dictConfig(config['logger'])
    logger = logging.getLogger()

    logging.info(f'child process id {os.getpid()}')
    logging.info(f'{nA}v{nB}_{now_time}')

    Xs = generate_Xs(nA=nA, nB=nB, AT_optionals=np.array([0.1, 1, 10]))
    #Ys_list = generate_Ys_list(dim=3)
    Ys_list = np.array([[1,0,0, 0,1,0, 0,0,1],
                        [0,0,1, 0,1,0, 1,0,0]])
    Ys_pred_list = []

    for i in tqdm(range(len(Ys_list))):
        Ys = Ys_list[i]

        Xs = torch.Tensor(Xs)
        Ys = torch.Tensor(Ys)
        train_ds = TensorDataset(Xs, Ys)
        train_dl = DataLoader(train_ds, batch_size=9, shuffle=False, drop_last=True)

        # 避免初始化的影响，重复训练3次
        loss_3 = []
        Ys_pred_3 = []
        for j in range(3):
            model = CompetitiveNetwork(nA, nB, nY, reparameterize='exp')
            device = torch.device("cpu")
            criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

            loss_epochs, Ys_pred = train(model, device, criterion, optimizer, train_dl, max_epochs, log_epochs, logger)
            
            loss_3.append(loss_epochs[-1])
            Ys_pred_3.append(Ys_pred)
            if (loss_epochs[-1] < 1e-2):
                break

        # 选择最佳的模型
        Ys_pred = Ys_pred_3[np.argmin(loss_3)]

        logger.info(f'final loss = {loss_epochs[-1]:.8f}')
        logger.info(f'Ys = {Ys}')
        logger.info(f'Ys_pred = {Ys_pred}')
        
        Ys_pred_list.append(Ys_pred)

    Ys_pred_list = np.array(Ys_pred_list)
    np.save(f'results/fix_A/2v{nB}_Ys_pred_list.npy', Ys_pred_list)



if __name__ == '__main__':
    
    config = parse_config('config.yaml')

    #main(config)
    #sys.exit(0)

    print(f'parent process id {os.getpid()}')
    pool = multiprocessing.Pool(processes=5)
    for nB in [2,3,4,5,6]:
        config1 = config.copy()
        config1['nB'] = nB
        pool.apply_async(main, (config1, ))
    pool.close()
    pool.join()

    print('All subprocesses done.')
