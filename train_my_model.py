import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from my_model import *

np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)

set_seed(42)

device = torch.device("cpu")
MAX_EPOCH = 1000


def generate_dataset(xmin, xmax, xstep):
    pass


if __name__ == '__main__':

    # ATs = np.array([[0.1, 0.1], [0.1, 1], [0.1, 10], [1, 0.1], [1, 1], [1, 10], [10, 0.1], [10, 1], [10, 10]], dtype=float)
    # BTs = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1,1]], dtype=float)

    a1 = np.array([0.1, 1, 10])
    ATs = np.array(list(itertools.product(a1, a1)))
    BTs = np.ones((3*3, 2))

    Xs = np.concatenate([ATs, BTs], axis=1)
    Ys = np.array([1,0,0, 0,1,0, 0,0,1], dtype=float)
    Xs = torch.Tensor(Xs)
    Ys = torch.Tensor(Ys)
    train_ds = TensorDataset(Xs, Ys)
    # train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    # model = CompetitiveNetwork_1(2, 2, 1).to(device)
    # model = CompetitiveNetwork_2(2, 2, 1).to(device)
    model = MLP(2, 2*2, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-2, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()

    loss_list = []
    for epoch in range(MAX_EPOCH):
        loss_sum = 0
        y_pred_list = []
        # 手动batch_size=4
        for i, (x, y) in enumerate(train_ds):
            #print(x, y)
            x = x.to(device)
            y = y.to(device)
            AT = x[:2]
            BT = x[2:]
            y_pred = model(AT, BT)

            #print(y)
            #print(y_pred)
            loss = loss_func(y, y_pred[0])
            loss_sum += loss
            y_pred_list.append(y_pred.detach().numpy())

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        loss_list.append(loss_sum.item())

        K = model.comp_layer.sqrt_K.data**2
        W = model.linear.weight.data
        B = model.linear.bias.data

        if (loss_sum.item() < 1e-3):
            break

        if (torch.max(K) > 100):
            break

        if (epoch % 100 == 0):
            print(f'epoch = {epoch}, loss = {loss_sum.item():.6f}')
            print(K)


    print('training completed')
    print(f'epoch = {epoch}, loss = {loss_sum.item():.6f}')

    plt.figure(figsize=(8,6), dpi=100)
    plt.plot(loss_list)
    plt.show()

    print('K', K)
    print('W', W)
    print('B', B)


    y_pred_mat = np.array(y_pred_list).reshape(3, 3)
    print('output', y_pred_mat)
    plot_heatmap(y_pred_mat)