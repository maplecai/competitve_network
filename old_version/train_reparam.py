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
from models.mlp import MLP
from utils.utils import *


def train(model, device, criterion, optimizer, dataloader, max_epochs, log_epochs, logger, train=True):
    logger.info(f'child pid = {os.getpid()}, start training one pattern')

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
            logger.debug(f'child pid = {os.getpid()}, epoch = {epoch}, loss = {loss_epochs[-1]:.8f}')

        if (epoch > 2*log_epochs):
        # loss基本不下降时结束训练
            if (np.mean(loss_epochs[-2*log_epochs:-log_epochs]) - np.mean(loss_epochs[-log_epochs:]) < 1e-4):
                break

    logger.info(f'child pid = {os.getpid()}, finish training one pattern, epoch = {epoch}, loss = {loss_epochs[-1]:.8f}')
    Ys_pred = np.array(Ys_pred).reshape(-1)

    return loss_epochs, Ys_pred




def main(config):
    seed = config['seed']
    set_seed(seed)
    nA = config['nA']
    nB = config['nB']
    nY = config['nY']
    reparam = config['model']['args']['reparam']
    mode = config['model']['args']['mode']
    task_time = config['task_time']
    task_name = f'{nA}_{nB}_{seed}_{reparam}_{mode}_{task_time}'

    log_dir = config['log_dir']
    output_dir = config['output_dir']
    max_epochs = config['max_epochs']
    log_epochs = config['log_epochs']

    config['logger']['handlers']['file_handler']['filename'] = f'{log_dir}/{task_name}.log'
    logging.config.dictConfig(config['logger'])
    logger = logging.getLogger()
    logger.info(f'parent pid = {os.getpid()}, start task {task_name}')
    logger.debug(config)

    Xs = generate_Xs(nA=nA, nB=nB, AT_optionals=np.array([0.1, 1, 10]))
    Ys_list = generate_Ys_list(dim=3)
    Ys_pred_list = []

    # 32核cpu
    # 当线程数设置为64时，单个进程cpu占用率50%，总cpu占用率40%，288训练时间
    pool = multiprocessing.Pool(processes=64)
    res_list = []
    
    for i in range(len(Ys_list)):
        Ys = Ys_list[i]

        Xs = torch.Tensor(Xs)
        Ys = torch.Tensor(Ys)
        train_ds = TensorDataset(Xs, Ys)
        train_dl = DataLoader(train_ds, batch_size=9, shuffle=False, drop_last=True)

        model = CompetitiveNetwork(nA, nB, nY, reparam=reparam, mode=mode)
        device = torch.device("cpu")
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

        res = pool.apply_async(train, (model, device, criterion, optimizer, train_dl, max_epochs, log_epochs, logger, ))
        res_list.append(res)

    pool.close()
    pool.join()

    for i in range(len(Ys_list)):
        loss_epochs, Ys_pred = res_list[i].get()
        Ys_pred_list.append(Ys_pred)
        
    Ys_pred_list = np.array(Ys_pred_list)
    np.save(f'{output_dir}/{task_name}.npy', Ys_pred_list)

    logger.info(f'parent pid = {os.getpid()}, finish task {task_name}, result save as {output_dir}/{task_name}.npy')



if __name__ == '__main__':
    
    config = parse_config('config.yaml')

    for reparam in ['exp', 'square']:
        for nB in [3,4,5]:

            task_time = datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')
            #if not os.path.exists(now_time):
            #    os.makedirs(now_time)

            config['task_time'] = task_time
            config['model']['args']['reparam'] = reparam
            config['nB'] = nB

            main(config)
