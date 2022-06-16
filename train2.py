import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from model2 import *

np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)

set_seed(42)

device = torch.device("cpu")
MAX_EPOCH = 10000
# BATCH_SIZE = 1

if __name__ == '__main__':
    ATs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    BTs = np.array([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=float)
    Xs = np.concatenate([ATs, BTs], axis=1)
    Ys = np.array([0, 1, 1, 0], dtype=float)
    Xs = torch.Tensor(Xs)
    Ys = torch.Tensor(Ys)
    train_ds = TensorDataset(Xs, Ys)
    # train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    # model = CompetitiveNetwork_1(2, 2, 1).to(device)
    model = CompetitiveNetwork_1(2, 2, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()

    loss_list = []
    for epoch in range(MAX_EPOCH):
        loss_sum = 0
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

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        loss_list.append(loss_sum.item())

        if (loss_sum.item() < 1e-3):
            break

        if (epoch % 100 == 0):
            print(f'epoch = {epoch}, loss = {loss_sum.item():.6f}')
            print(model.comp_layer.sqrt_K.data**2)


    print('training completed')
    print(f'epoch = {epoch}, loss = {loss_sum.item():.6f}')
    print(model.comp_layer.sqrt_K.data**2)
    plt.figure(figsize=(8,6), dpi=100)
    plt.plot(loss_list)
    plt.show()
