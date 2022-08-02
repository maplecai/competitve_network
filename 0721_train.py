import os
import sys
import time
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
from utils.utils import *


def train(config, Xs, Ys, logger, train=True):
    seed = config['seed']
    set_seed(seed)

    nA = config['nA']
    nB = config['nB']
    nY = config['nY']
    reparam = config['model']['args']['reparam']
    mode = config['model']['args']['mode']

    max_epochs = config['max_epochs']
    log_epochs = config['log_epochs']
    tol = config['tol']

    # Ys_str = ''.join(str(int(i)) for i in Ys)
    pid = os.getpid()
    logger.info(f'child pid = {pid}, start training {Ys}')

    train_ds = TensorDataset(torch.Tensor(Xs), torch.Tensor(Ys))
    train_dl = DataLoader(train_ds, batch_size=9, shuffle=False, drop_last=True)

    model = CompetitiveNetwork(nA, nB, nY, reparam=reparam, mode=mode, trainBT=True)
    device = torch.device("cpu")
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

    model.to(device)
    model.train()

    loss_epochs = []
    for epoch in range(max_epochs+1):
        loss_epoch = 0
        Ys_pred = []

        for i, (x, y) in enumerate(train_dl):
            x = x.to(device)
            y = y.to(device)
            y_pred = torch.Tensor(y.shape).to(device)
            for j in range(len(x)):
                # 模型一次只能输入一个样本
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

        '''if (epoch % log_epochs == 0):
            logger.debug(f'child pid = {pid}, epoch = {epoch}, loss = {loss_epochs[-1]:.8f}')

            K = model.comp_layer.K.data
            BT = model.comp_layer.BT.data
            logger.debug(f'child pid = {pid}, K = {K}, BT = {BT}')'''

        if (epoch > 2*log_epochs):
        # 过去 log_epcohs 的平均loss基本不下降时结束训练
            if (np.mean(loss_epochs[-2*log_epochs:-log_epochs]) - np.mean(loss_epochs[-log_epochs:]) < tol):
                break

        # 因为训练BT时很容易模式崩溃，一有上升迹象就立刻停止训练
        '''if (epoch > log_epochs):
            if (loss_epochs[-1] > loss_epochs[-2]):
                break'''

    logger.debug(f'child pid = {pid}, finish training {Ys}, epoch = {epoch}, loss = {loss_epochs[-1]:.8f}')
    if np.isnan(loss_epochs).any() == True:
        logger.warning(f'child pid = {pid}, finish training {Ys}, epoch = {epoch}, loss = {loss_epochs[-1]:.8f}')
    
    Ys_pred = np.array(Ys_pred).reshape(-1)

    return loss_epochs, Ys_pred



def main(config):
    seed = config['seed']
    nA = config['nA']
    nB = config['nB']
    nY = config['nY']
    reparam = config['model']['args']['reparam']
    mode = config['model']['args']['mode']
    log_dir = config['log_dir']
    figure_dir = config['figure_dir']
    output_dir = config['output_dir']

    task_time = datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')
    task_name = f'{nA}_{nB}_{mode}_{reparam}_{seed}'

    config['logger']['handlers']['file_handler']['filename'] = f'{log_dir}/{task_time}_{task_name}.log'
    logging.config.dictConfig(config['logger'])
    logger = logging.getLogger()
    logger.info(f'running {__file__}')
    logger.info(f'parent pid = {os.getpid()}, start task {task_name}')
    logger.debug(config)

    ATs = generate_ATs(nA=nA)
    BTs = generate_BTs(nB=nB)
    Xs = np.concatenate([ATs, BTs], axis=1)
    Ys_list = generate_Ys_list(dim=3)
    
    pool = multiprocessing.Pool(processes=30) #32核服务器
    res_list = []
    for i in range(len(Ys_list)):
        Ys = Ys_list[i]
        res = pool.apply_async(train, (config, Xs, Ys, logger, ))
        res_list.append(res)
    pool.close()
    pool.join()

    Ys_pred_list = []
    loss_epochs_list = []
    for i in range(len(Ys_list)):
        loss_epochs, Ys_pred = res_list[i].get()
        Ys_pred_list.append(Ys_pred)
        loss_epochs_list.append(loss_epochs)

    '''for i in range(len(Ys_list)):
        plt.plot(loss_epochs_list[i])
        plt.xlim([0, 1000])
        plt.savefig(f'{figure_dir}/{i}.png')
        plt.close()'''
        
    Ys_pred_list = np.array(Ys_pred_list)
    np.save(f'{output_dir}/{task_name}_{task_time}.npy', Ys_pred_list)

    logger.info(f'parent pid = {os.getpid()}, finish task {task_name}')



if __name__ == '__main__':
    
    config = parse_config('config.yaml')

    for nB in [2]:
        for mode in ['fixA']:
            for reparam in ['square']:
                for seed in [0]:
                    config['nB'] = nB
                    config['seed'] = seed
                    config['model']['args']['reparam'] = reparam
                    config['model']['args']['mode'] = mode

                    main(config)

