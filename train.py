import argparse
import datetime
import itertools
import logging
import logging.config
import os
import sys
import time
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
#import multiprocessing
import torch.nn as nn
import torch.utils.data
import yaml

import models
from utils import *

multiprocessing.set_sharing_strategy('file_system')

def train(config, X, Y, logger):
    # set random seed
    set_seed(config['seed'])

    # create model
    device = torch.device(config['device'])
    model = init_obj(models, config['model']).to(device)
    loss_func = init_obj(torch.nn, config['loss_func'])
    #reg_loss_func = init_obj(models, config['reg_loss_func'])
    optimizer = init_obj(torch.optim, config['optimizer'], model.parameters())
    early_stopping = init_obj(utils, config['early_stopping'])

    train_dataset = torch.utils.data.TensorDataset(X, Y)
    valid_dataset = torch.utils.data.TensorDataset(X, Y)
    train_dataloader = init_obj(torch.utils.data, config['data_loader'], train_dataset)
    valid_dataloader = init_obj(torch.utils.data, config['data_loader'], valid_dataset)
    train_loss_list = []
    valid_loss_list = []
    
    # train
    for epoch in range(config['max_epoch_num']):
        model.train()
        loss_list = []
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = loss_func(out, y)
            #loss += reg_loss_func(model)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if model.constrain == 'clamp':
            #     model.clamp_parameters(1e-6, 1e6)
            
        train_loss = np.mean(loss_list)
        train_loss_list.append(train_loss)
        logger.debug(f'epoch = {epoch:3}, train loss = {train_loss:.6f}')

        # # valid
        # if (epoch % config['valid_epoch_num'] == 0):
        #     with torch.no_grad():
        #         model.eval()
        #         loss_list = []
        #         for i, (x, y) in enumerate(valid_dataloader):
        #             x = x.to(device)
        #             y = y.to(device)
        #             out = model(x)
        #             loss = loss_func(out, y)
        #             #loss += reg_loss_func(model)

        #             loss_list.append(loss.item())

        #         valid_loss = np.mean(loss_list)
        #         valid_loss_list.append(valid_loss)
        #         #logger.debug(f'epoch = {epoch:3}, valid loss = {valid_loss:.6f}')
        #     #break
        valid_loss = train_loss
        early_stopping.check(valid_loss, model.state_dict())
        if early_stopping.flag is True:
            break
            #break

    logger.debug(f'epoch = {epoch:3}, stop training, best valid loss = {valid_loss:.6f}')

    result = {'config': config, 
              'best_valid_loss': early_stopping.best_valid_loss,
              'best_model_state_dict': pickle.dumps(early_stopping.best_model_state_dict),
              #'best_model_state_dict': deepcopy(early_stopping.best_model_state_dict),
              #'best_model_state_dict': model.state_dict(),
              #'train_loss_list': train_loss_list,
              #'valid_loss_list': valid_loss_list,
              }
    return result


def main(config):

    task_name = config['task_name']
    log_dir = config['log_dir']
    results_dir = config['results_dir']

    now_time = datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S_%f')
    model_args = '_'.join([str(v) for v in config['model']['args'].values()])
    file_name = f'{now_time}_{task_name}_{model_args}'
    config['logger']['handlers']['file_handler']['filename'] = f'{log_dir}/{file_name}.log'

    logging.config.dictConfig(config['logger'])
    logger = logging.getLogger()
    logger.debug(f'{file_name} start')
    logger.debug(yaml.dump(config))

    X = np.load(config['data_dir'] + config['x_filename'])
    Y_list = np.load(config['data_dir'] + config['y_filename'])
    X = torch.tensor(X, dtype=torch.float)
    if Y_list.dtype is np.dtype('int64'):
        Y_list = torch.tensor(Y_list, dtype=torch.long)
    else:
        Y_list = torch.tensor(Y_list, dtype=torch.float)

    result_array = []

    for seed in range(config['train_times']):
        config['seed'] = seed

        if config.get('multiprocess', True) is True:
            pool = multiprocessing.Pool(processes=config['process_num'])
            process_list = []
            for i in range(len(Y_list)):
                process = pool.apply_async(train, (config.copy(), X, Y_list[i], logger))
                process_list.append(process)
            pool.close()
            pool.join()
            result_list = [process.get() for process in process_list]
        else:
            result_list = [train(config.copy(), X, Y_list[i], logger) for i in range(len(Y_list))]

        logger.debug(f'finish training seed = {seed}')
        result_array.append(result_list)

    result_array = np.array(result_array).T
    np.save(f'{results_dir}/{file_name}', result_array)

    logger.info(f'{file_name} finish')

    return result_list


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args = args.parse_args()
    config = parse_config(args.config)


    results = main(config)





    # for (config['model']['args']['nB'], 
    #         config['model']['args']['mode'], 
    #         config['model']['args']['output'], 
    #         config['model']['args']['linear_constrain']) \
    #     in itertools.product(
    #         [1,2,3,4],
    #         ['comp'],
    #         ['ABC'],
    #         ['none']):
    #     results = main(config)

    # for (config['model']['args']['nB'], 
    #         config['model']['args']['mode'], 
    #         config['model']['args']['output'], 
    #         config['model']['args']['linear_constrain']) \
    #     in itertools.product(
    #         [1,2,3,4],
    #         ['comp', 'semiA', 'semiB'],
    #         ['ABC'],
    #         ['none']):
    #     results = main(config)

    # for (config['model']['args']['nB'], 
    #         config['model']['args']['mode'], 
    #         config['model']['args']['output'], 
    #         config['model']['args']['linear_constrain']) \
    #     in itertools.product(
    #         [1,2,3,4],
    #         ['comp'],
    #         ['ABC', 'A', 'B', 'C', 'C11'],
    #         ['none']):
    #     results = main(config)

    # for (config['model']['args']['nB'], 
    #         config['model']['args']['mode'], 
    #         config['model']['args']['output'], 
    #         config['model']['args']['linear_constrain']) \
    #     in itertools.product(
    #         [1,2,3,4],
    #         ['comp'],
    #         ['ABC', 'C'],
    #         ['positive']):
    #     results = main(config)

