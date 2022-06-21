import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from models import *

np.set_printoptions(suppress=True)

set_seed(42)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
MAX_EPOCH = 10000
BATCH_SIZE = 4
tol = 1e-3
loss_func = nn.MSELoss()
loss_list = []


if __name__ == '__main__':
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    train_ds = TensorDataset(x, y)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TwoLayerNet(2, 2, 1).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    for epoch in range(MAX_EPOCH):
        for batch_id, (x, y) in enumerate(train_dl):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = loss_func(out.reshape(-1), y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        if (epoch % 100 == 0):
            print(f'epoch = {epoch}, loss = {loss.item()}')

        if (loss.item() < 1e-3):
            break

    print('K', model.K.detach().numpy() ** 2)
    print('u', model.u.detach().numpy())
    print('b', model.b.detach().numpy())
    print('y', out.detach().numpy())
    
    plt.figure(figsize=(8,6), dpi=100)
    plt.plot(loss_list)
    plt.show()
