import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils import *
from models import *


np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)

set_seed(42)


def train(model, device, criterion, optimizer, dataloader, dataset, max_epochs, log_steps):

    model.to(device)
    model.train()
    loss_list = []

    for epoch in range(max_epochs+1):
        loss_sum = 0
        y_pred_list = []
        # 手动batch
        for i, (x, y) in enumerate(dataset):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x[:2], x[2:])

            #print(y)
            #print(y_pred)
            loss = criterion(y, y_pred[0])
            loss_sum += loss
            y_pred_list.append(y_pred.detach().numpy())

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        loss_list.append(loss_sum.item())

        K = model.comp_layer.param
        W = model.linear.weight.data
        B = model.linear.bias.data

        # print(loss_sum.item())

        #if (epoch > log_steps):
        #    if (np.mean(loss_list[-2*log_steps:-log_steps]) - np.mean(loss_list[-log_steps:]) < 1e-3):
        #        break

        if (epoch % log_steps == 0):
            print(f'epoch = {epoch}, loss = {loss_sum.item():.6f}')
            #print('K', K.data)
            #print('W', W.data)
            #print('B', B.data)

    # plt.figure(figsize=(8,6), dpi=100)
    # plt.plot(loss_list)
    # plt.show()

    return y_pred_list




def generate_data():

    possible_Ys = list(itertools.product([0,1], repeat=3))
    a = possible_Ys[0]
    print(a)
    b = [str(i) for i in a]
    b=''.join(b)
    print(b)



def main():

    y_pred_array = []
    at = np.array([0.1, 1, 10])
    ATs = np.array(list(itertools.product(at, at)))
    BTs = np.ones((3*3, 2))

    Xs = np.concatenate([ATs, BTs], axis=1).astype(float)
    possible_Ys = list(itertools.product([0, 1], repeat=9))

    for Ys in tqdm(possible_Ys):
        print(Ys)
        fig_name = [str(i) for i in Ys]
        fig_name = ''.join(fig_name)

        Ys = np.array(Ys).astype(float)
        Xs = torch.Tensor(Xs)
        Ys = torch.Tensor(Ys)
        train_ds = TensorDataset(Xs, Ys)
        train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)

        model = CompetitiveNetwork(2, 2, 1, reparameterize='square', gradient='linear_algebra')
        device = torch.device("cpu")
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        max_epochs = 1000
        log_steps = 100

        y_pred_list = train(model, device, criterion, optimizer, train_dl, train_ds, max_epochs, log_steps)

        y_pred_mat = np.array(y_pred_list).reshape(3, 3)
        print('output', y_pred_mat)
        plot_heatmap(y_pred_mat, x=at, fig_name=fig_name)

        y_pred_array.append(y_pred_list.reshape(-1))

    y_pred_array = np.array(y_pred_array)
    np.save('y_pred_array.npy', y_pred_array)


if __name__ == '__main__':
    main()
