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

        return AF, BF


if __name__ == '__main__':
    solver = Solver(device=torch.device("cpu"), batch_size=1, max_iter=100, tol=1e-6)

    AT = torch.Tensor([[1]])
    BT = torch.Tensor([[1]])
    K  = torch.Tensor([[1]])
    AF, BF = solver.solve(AT, BT, K)
    print(AF, BF)
    C = AF * BF * K

    for dK in [1e-1, 1e-2, 1e-3, 1e-4]:
        K1 = K + dK
        AF1, BF1 = solver.solve(AT, BT, K1)
        
        C1 = AF1 * BF1 * K1
        pC = (C1 - C) / dK
        print(pC)

    ans = AF * BF / (1 + K*AF +K*BF)
    print(f'theoretical answer is {ans}'.format(ans=ans))


'''
    AT = torch.Tensor([[1, 1]])
    BT = torch.Tensor([[1, 1]])
    K  = torch.Tensor([[1, 1], [1, 1]])
    AF, BF = solver.solve(AT, BT, K)
    print(AF, BF)
'''