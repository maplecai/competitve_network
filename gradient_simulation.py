import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

# torch version
class Solver():
    def __init__(self, device, batch_size, max_iter, tol):
        self.device = device
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        
    '''
    def torch_solve_batch(self, AT: torch.Tensor, BT: torch.Tensor, K:torch.Tensor):
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
    '''

    def torch_solve(self, AT: torch.Tensor, BT: torch.Tensor, K:torch.Tensor):
        # AT.shape = (NA, 1)
        # BT.shape = (1, NB)
        AF = torch.zeros(AT.shape)
        BF = torch.zeros(BT.shape)

        with torch.no_grad():
            err = 0
            for iter in range(self.max_iter):
                AF_last = AF
                BF_last = BF
                AF = AT / ((K * BF_last).sum(axis=1, keepdims=True) + 1)
                BF = BT / ((K * AF_last).sum(axis=0, keepdims=True) + 1)
                # rint('iter', iter, AF, BF)
                err = np.abs(AF-AF_last).sum() + np.abs(BF-BF_last).sum()
                if (err < self.tol):
                    # print(f'converge in {iter} iterations')
                    break
            if (err > self.tol):
                print(f'not converge in {iter} iterations')
            
            C = K * AF * BF
        return AF, BF, C


    def torch_autograd(self, AF, BF, K):
        # simultaneous iteration
        AF_last = AF
        BF_last = BF
        AF = AT / ((K * BF_last).sum(axis=1, keepdims=True) + 1)
        BF = BT / ((K * AF_last).sum(axis=0, keepdims=True) + 1)
        C = K * AF * BF

        pC = torch.zeros(K.shape)
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                C[i, j].backward(retain_graph=True)
                pC[i, j] = K.grad[0, 0]
                K.grad.detach_()
                K.grad.zero_()
                # print(pC[i, j])

        return pC


    def torch_autograd2(self, AF, BF, K):
        # alternative iteration
        AF = AT / ((K * BF).sum(axis=1, keepdims=True) + 1)
        BF = BT / ((K * AF).sum(axis=0, keepdims=True) + 1)
        C = K * AF * BF

        pC = torch.zeros(K.shape)
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                C[i, j].backward(retain_graph=True)
                pC[i, j] = K.grad[0, 0]
                K.grad.detach_()
                K.grad.zero_()
                # print(pC[i, j])

        return pC



    def np_solve_(self, AT: np.ndarray, BT: np.ndarray, K:np.ndarray):
        # no batch
        # K.shape = (nA, nB)
        nA = len(AT)
        nB = len(BT)
        AT = AT.reshape(nA, 1)
        BT = BT.reshape(1, nB)
        AF, BF, C = self.np_solve(AT, BT, K)
        AF = AF.reshape(nA)
        BF = BF.reshape(nB)
        return AF, BF, C


    def np_solve(self, AT: np.ndarray, BT: np.ndarray, K:np.ndarray):
        # no batch
        # AT.shape = (NA, 1)
        # BT.shape = (1, NB)
        AF = np.zeros(AT.shape)
        BF = np.zeros(BT.shape)

        err = 0
        for iter in range(self.max_iter):
            AF_last = AF
            BF_last = BF
            AF = AT / ((K * BF).sum(axis=1, keepdims=True) + 1)
            BF = BT / ((K * AF).sum(axis=0, keepdims=True) + 1)
            # print('iter', iter, AF, BF)
            err = np.abs(AF-AF_last).sum() + np.abs(BF-BF_last).sum()
            if (err < self.tol):
                break
        if (err > self.tol):
            print(f'not converge in {iter} iterations')

        C = K * AF * BF
        return AF, BF, C



    def calc_partial_derivative(self, AF, BF, K, p=0, q=0):
        # 先求解pA/pK, pB/pK, nA+nB维线性方程组
        nA = len(AF)
        nB = len(BF)
        W = np.zeros((nA+nB, nA+nB))
        b = np.zeros((nA+nB, 1))
        # 一行一行填
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
        
        pApB = np.linalg.solve(W, b)
        # print(pApB)

        # 然后求pC/pK
        pC = np.zeros((nA, nB))
        for i in range(nA):
            for j in range(nB):
                pC[i, j] += K[i, j] * BF[j] * pApB[i] + K[i, j] * AF[i] * pApB[nA+j]
        pC[p, q] += AF[p] * BF[q]

        return pC



    def manual_2v2_partial_derivative(self, AF, BF, K):
        W = np.array([[1 + K[0,0]*AF[0] + K[0,0]*BF[0], K[0,0]*BF[0], K[0,0]*AF[0], 0],
                    [K[0,1]*BF[1], 1 + K[0,1]*BF[1] + K[0,1]*AF[0], 0, K[0,1]*AF[0]],
                    [K[1,0]*AF[1], 0, 1 + K[1,0]*AF[1] + K[1,0]*BF[0], K[1,0]*BF[0]],
                    [0, K[1,1]*AF[1], K[1,1]*BF[1], 1 + K[1,1]*AF[1] + K[1,1]*BF[1]]])
        b = np.array([[AF[0]*BF[0]], [0], [0], [0]])
        pC = np.linalg.solve(W, b)
        pC = pC.reshape(2,2)
        return pC



# numpy version
if __name__ == '__main__':

    # 2v2 competition
    solver = Solver(device=torch.device("cpu"), batch_size=1, max_iter=100, tol=1e-12)
    AT = np.array([1., 1.])
    BT = np.array([1., 1., 1.])
    K  = np.array([1., 2., 3., 4., 5., 6.]).reshape(2, 3)
    AF, BF, C = solver.np_solve_(AT, BT, K)
    print('AT, BT, K')
    print(AT, BT, K)
    print('AF, BF, C')
    print(AF, BF, C)

    # numerical simulation
    for dK11 in [1e-4]:
        AT_ = AT
        BT_ = BT
        K_ = K.copy()
        K_[0, 0] = K[0, 0]+ dK11
        # print('K = ', K_)
        AF_, BF_, C_ = solver.np_solve_(AT_, BT_, K_)

        # print(AF_, BF_, C_)
        pC1 = (C_ - C) / dK11
    print('numerical simulation pC/pK00')
    print(pC1)

    # manual
    pC2 = solver.manual_2v2_partial_derivative(AF, BF, K)
    print('my manual pC/pK00')
    print(pC2)

    # theoretical
    pC3 = solver.calc_partial_derivative(AF, BF, K)
    print('my function pC/pK00')
    print(pC3)


    # torch autograd
    AT = torch.Tensor([1, 1]).reshape(2,1)
    BT = torch.Tensor([1, 1, 1]).reshape(1,3)
    K  = torch.Tensor([1, 2, 3, 4, 5, 6]).reshape(2,3)
    K.requires_grad = True

    #print(K.grad)
    #y = torch.sum(K*2)
    #y.backward()
    #print(K.grad)

    AF, BF, C = solver.torch_solve(AT, BT, K)
    # print(AF, BF, C)
    
    pC4 = solver.torch_autograd(AF, BF, K)
    print('torch autograd (simultaneous) pC/pK00')
    print(pC4)

    pC5 = solver.torch_autograd2(AF, BF, K)
    print('torch autograd (alternative) pC/pK00')
    print(pC5)

