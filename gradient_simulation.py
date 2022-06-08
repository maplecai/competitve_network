import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

# torch version
class Solver():
    def __init__(self, device="cpu", max_iter=100, tol=1e-6):
        self.device = device
        self.max_iter = max_iter
        self.tol = tol



    def np_solve(self, AT: np.ndarray, BT: np.ndarray, K:np.ndarray):
        # 求解
        nA = len(AT)
        nB = len(BT)
        AF = np.zeros(AT.shape)
        BF = np.zeros(BT.shape)

        err = 0
        for iter in range(self.max_iter):
            AF_last = AF
            BF_last = BF
            AF = AT / ((K * BF.reshape(1, nB)).sum(axis=1) + 1)
            BF = BT / ((K * AF.reshape(nA, 1)).sum(axis=0) + 1)
            # print('iter', iter, AF, BF)
            err = np.abs(AF-AF_last).sum() + np.abs(BF-BF_last).sum()
            if (err < self.tol):
                break
        if (err > self.tol):
            print(f'not converge in {iter} iterations')
        C = K * AF.reshape(nA, 1) * BF.reshape(1, nB)

        return AF, BF, C


    def np_gradient(self, AF, BF, K, p=0, q=0):
        # 求解dC/dKpq
        # 先求解dAF/dKpq, dBF/dKpq, nA+nB维线性方程组
        nA = len(AF)
        nB = len(BF)
        W = np.zeros((nA+nB, nA+nB))
        b = np.zeros((nA+nB, 1))
        # 一行一行填W b
        for m in range(nA):
            # 第m行
            for j in range(nB):
                # 第nA+j列
                W[m, nA+j] += K[m, j] * AF[m]
                # 第m列
                W[m, m] += K[m, j] * BF[j]
            W[m, m] += 1
        for n in range(nB):
            # 第nA+n行
            for i in range(nA):
                # 第i列
                W[nA+n, i] += K[i, n] * BF[n]
                # 第nA+n列
                W[nA+n, nA+n] += K[i, n] * AF[i]
            W[nA+n, nA+n] += 1
        b[p, 0] += - AF[p] * BF[q]
        b[nA+q, 0] += - AF[p] * BF[q]
        
        dAdB = np.linalg.solve(W, b)
        dA = dAdB[:nA]
        dB = dAdB[nA:]

        # 然后求dC/pKpq
        dC = np.zeros((nA, nB))
        for i in range(nA):
            for j in range(nB):
                dC[i, j] += K[i, j] * BF[j] * dA[i] + K[i, j] * AF[i] * dB[j]
        dC[p, q] += AF[p] * BF[q]

        return dC


    '''
    def np_gradient_2v2_manual(self, AF, BF, K):
        # 手推的直接解dC的方程
        W = np.array([[1 + K[0,0]*AF[0] + K[0,0]*BF[0], K[0,0]*BF[0], K[0,0]*AF[0], 0],
                    [K[0,1]*BF[1], 1 + K[0,1]*BF[1] + K[0,1]*AF[0], 0, K[0,1]*AF[0]],
                    [K[1,0]*AF[1], 0, 1 + K[1,0]*AF[1] + K[1,0]*BF[0], K[1,0]*BF[0]],
                    [0, K[1,1]*AF[1], K[1,1]*BF[1], 1 + K[1,1]*AF[1] + K[1,1]*BF[1]]])
        b = np.array([[AF[0]*BF[0]], [0], [0], [0]])
        dC = np.linalg.solve(W, b)
        dC = dC.reshape(2,2)
        return dC
    '''


    def torch_solve(self, AT: np.ndarray, BT: np.ndarray, K:np.ndarray):
        # 求解
        nA = len(AT)
        nB = len(BT)
        AF = torch.zeros(AT.shape)
        BF = torch.zeros(BT.shape)

        with torch.no_grad():
            err = 0
            for iter in range(self.max_iter):
                AF_last = AF
                BF_last = BF
                AF = AT / ((K * BF.reshape(1, nB)).sum(axis=1) + 1)
                BF = BT / ((K * AF.reshape(nA, 1)).sum(axis=0) + 1)
                # print('iter', iter, AF, BF)
                err = np.abs(AF-AF_last).sum() + np.abs(BF-BF_last).sum()
                if (err < self.tol):
                    break
            if (err > self.tol):
                print(f'not converge in {iter} iterations')
            C = K * AF.reshape(nA, 1) * BF.reshape(1, nB)

        return AF, BF, C




    def torch_autograd(self, AF, BF, K):
        # pytorch 自动求导最后一步迭代
        nA = len(AF)
        nB = len(BF)
        AF_last = AF
        BF_last = BF
        # simultaneous iteration
        AF = AT / ((K * BF_last.reshape(1, nB)).sum(axis=1) + 1)
        BF = BT / ((K * AF_last.reshape(nA, 1)).sum(axis=0) + 1)
        # or alternative iteration
        # AF = AT / ((K * BF).sum(axis=1, keepdims=True) + 1)
        # BF = BT / ((K * AF).sum(axis=0, keepdims=True) + 1)
        C = K * AF.reshape(nA, 1) * BF.reshape(1, nB)

        dC = torch.zeros(K.shape)
        for i in range(nA):
            for j in range(nB):
                C[i, j].backward(retain_graph=True)
                dC[i, j] = K.grad[0, 0]
                K.grad.detach_()
                K.grad.zero_()
                # print(dC[i, j])

        return dC





if __name__ == '__main__':

    # 2v2 competition
    solver = Solver()
    AT = np.array([1., 1.])
    BT = np.array([1., 1., 1.])
    K  = np.array([1., 2., 3., 4., 5., 6.]).reshape(2, 3)
    AF, BF, C = solver.np_solve(AT, BT, K)
    print('AT, BT, K')
    print(AT, BT, K)
    print('AF, BF, C')
    print(AF, BF, C)

    # numerical simulation
    for dK00 in [1e-4]:
        AT_ = AT
        BT_ = BT
        K_ = K.copy()
        K_[0, 0] = K[0, 0]+ dK00
        # print('K = ', K_)
        AF_, BF_, C_ = solver.np_solve(AT_, BT_, K_)

        # print(AF_, BF_, C_)
        dC1 = (C_ - C) / dK00
    print('numerical simulation dC/pK00')
    print(dC1)

    # manual
    # dC2 = solver.manual_2v2_dArtial_derivative(AF, BF, K)
    # print('my manual dC/pK00')
    # print(dC2)

    # theoretical
    dC3 = solver.np_gradient(AF, BF, K)
    print('my linalg function dC/pK00')
    print(dC3)

    


    # torch autograd
    AT = torch.Tensor([1, 1])
    BT = torch.Tensor([1, 1, 1])
    K  = torch.Tensor([1, 2, 3, 4, 5, 6]).reshape(2,3)
    K.requires_grad = True

    AF, BF, C = solver.torch_solve(AT, BT, K)
    # print(AF, BF, C)

    dC4 = solver.torch_autograd(AF, BF, K)
    print('torch autograd (simultaneous) dC/pK00')
    print(dC4)


    with torch.no_grad():
        dC5 = solver.np_gradient(AF, BF, K)
        print(dC5)

