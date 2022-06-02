import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# torch version
class Solver():
    def __init__(self, device, batch_size, max_iter, tol):
        self.device = device
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, AT: torch.Tensor, BT: torch.Tensor, K:torch.Tensor):
        AF = torch.zeros((self.batch_size, K.shape[0], 1)).to(self.device)
        BF = torch.zeros((self.batch_size, K.shape[1], 1)).to(self.device)

        with torch.no_grad():
            err = 0
            for iter in range(self.max_iter):
                AF_last = AF
                BF_last = BF
                AF = torch.unsqueeze(AT, dim=2) / (K.matmul(BF)+1)
                BF = torch.unsqueeze(BT, dim=1) / (K.T.matmul(AF)+1)
                err = (AF-AF_last).abs().max() + (BF-BF_last).abs().max()
                if (err < self.tol):
                    break
            if (err > self.tol):
                print(f'not converge in {iter} iterations')

            C = K * AF.matmul(BF.transpose(1, 2))
        return AF, BF, C


    def np_solve(self, AT: np.ndarray, BT: np.ndarray, K:np.ndarray):
        # batch_size = 1
        # AT.shape = (NA, 1)
        # BT.shape = (1, NB)
        AF = np.zeros(AT.shape)
        BF = np.zeros(BT.shape)

        with torch.no_grad():
            err = 0
            for iter in range(self.max_iter):
                AF_last = AF
                BF_last = BF
                AF = AT / ((K * BF).sum(axis=1, keepdims=True) + 1)
                BF = BT / ((K * BF).sum(axis=0, keepdims=True) + 1)
                err = np.abs(AF-AF_last).sum() + np.abs(BF-BF_last).sum()
                if (err < self.tol):
                    break
            if (err > self.tol):
                print(f'not converge in {iter} iterations')

            C = K * AF * BF
        return AF, BF, C



if __name__ == '__main__':
    # numpy version

    solver = Solver(device=torch.device("cpu"), batch_size=1, max_iter=100, tol=1e-9)
    # 2v2
    AT = np.array([1., 1.]).reshape(2, 1)
    BT = np.array([1., 1.]).reshape(1, 2)
    K  = np.array([1., 1., 1., 1.]).reshape(2, 2)
    AF, BF, C = solver.np_solve(AT, BT, K)
    print(AF, BF, C)

    for dK11 in [1e-1, 1e-2, 1e-3, 1e-4]:
        AT_ = AT
        BT_ = BT
        K_ = K.copy()
        K_[0, 0] = K[0, 0]+ dK11
        print('K = ', K_)
        AF_, BF_, C_ = solver.np_solve(AT_, BT_, K_)

        # print(AF_, BF_, C_)
        
        C_ = K_ * AF_ * BF_
        print(C_)
        pC = (C_ - C) / dK11
        print(f'numerical simulation partial C / partial K11 = ', pC)







'''

if __name__ == '__main__':

    solver = Solver(device=torch.device("cpu"), batch_size=1, max_iter=100, tol=1e-6)

    # 1v1
    AT = torch.Tensor([[1]])
    BT = torch.Tensor([[1]])
    K  = torch.Tensor([[1]])
    AF, BF = solver.solve(AT, BT, K)
    print(AF, BF)
    C = AF * BF * K

    for dK in [1e-1, 1e-2, 1e-3, 1e-4]:
        K1 = K + dK
        AF1, BF1, C1 = solver.solve(AT, BT, K1)
        
        C1 = AF1 * BF1 * K1
        pC = (C1 - C) / dK
        print(pC)

    ans = AF * BF / (1 + K*AF +K*BF)
    print(f'my theoretical answer is {ans}'.format(ans=ans))


'''