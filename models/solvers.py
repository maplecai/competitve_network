import numpy as np
import torch
import time
import eqtk


torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)


def numpy_solve(AT, BT, K, max_iter=30, tol=1e-3):
    '''
    Using numpy to solve the equilibrium
    '''
    nA = len(AT)
    nB = len(BT)
    AF = np.zeros(AT.shape, dtype=np.float32)
    BF = np.zeros(BT.shape, dtype=np.float32)

    err = 0
    for iter in range(max_iter):
        AF_last = AF
        BF_last = BF
        AF = AT / ((K * BF.reshape(1, nB)).sum(axis=1) + 1)
        BF = BT / ((K * AF.reshape(nA, 1)).sum(axis=0) + 1)
        # print('iter', iter, AF, BF)
        err = np.abs(AF-AF_last).sum() + np.abs(BF-BF_last).sum()
        if (err < tol):
            break
    if (err > tol):
        print(f'not converge in {max_iter} iterations')
    C = K * AF.reshape(nA, 1) * BF.reshape(1, nB)

    return AF, BF, C



def numpy_gradient(AF, BF, K):
    '''
    Using numpy to solve the gradient
    '''
    nA = len(AF)
    nB = len(BF)
    dC_dK = np.zeros((nA, nB, nA, nB), dtype=np.float32)
    dA_dK = np.zeros((nA, nA, nB), dtype=np.float32)
    dB_dK = np.zeros((nB, nA, nB), dtype=np.float32)
    
    
    # 先求解dAF/dKpq, dBF/dKpq, 解nA+nB维线性方程组
    W = np.zeros((nA+nB, nA+nB), dtype=np.float32)
    # 一行一行填W
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

    W_inv = np.linalg.inv(W)
    
    # t0 = time.perf_counter()
    # for i in range(1000):
    #     W_inv = np.linalg.inv(W)
    # t1 = time.perf_counter()
    # print('process time', t1-t0)

    # 对于给定的Kpq
    for p in range(nA):
        for q in range(nB):
            b = np.zeros((nA+nB, 1))
            b[p, 0] += - AF[p] * BF[q]
            b[nA+q, 0] += - AF[p] * BF[q]
            x = np.matmul(W_inv, b).reshape(-1)
            dA_dK[:, p, q] = x[:nA]
            dB_dK[:, p, q] = x[nA:]

    # 然后求dC/pKpq
    for i in range(nA):
        for j in range(nB):
            dC_dK[i, j] += K[i, j] * BF[j] * dA_dK[i] + K[i, j] * AF[i] * dB_dK[j]
            dC_dK[i, j, i, j] += AF[i] * BF[j]

    return dC_dK



def torch_solve(AT, BT, K, max_iter=20, tol=1e-3):
    '''
    Using torch to solve the equilibrium
    '''
    nA = len(AT)
    nB = len(BT)
    AF = torch.zeros(AT.shape)
    BF = torch.zeros(BT.shape)

    err = 0
    for iter in range(max_iter):
        AF_last = AF
        BF_last = BF
        AF = AT / ((K * BF.reshape(1, nB)).sum(axis=1) + 1)
        BF = BT / ((K * AF.reshape(nA, 1)).sum(axis=0) + 1)
        # print('iter', iter, AF, BF)
        err = torch.abs(AF-AF_last).sum() + torch.abs(BF-BF_last).sum()
        if (err < tol):
            break
    if (err > tol):
        print(f'not converge in {iter} iterations')
    C = K * AF.reshape(nA, 1) * BF.reshape(1, nB)

    return AF, BF, C



def torch_iterate_last(AT, BT, K, AF, BF, C):
    '''
    Using torch to iterate the last time to get torch.autograd
    '''
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

    return AF, BF, C



def eqtk_solve(AT, BT, K):
    nA = len(AT)
    nB = len(BT)

    c0 = np.concatenate([AT, BT, np.zeros(nA*nB)])
    N = np.zeros((nA*nB, nA + nB + nA*nB))
    for i in range(nA):
        for j in range(nB):
            k = nB*i + j
            N[k, i] = -1
            N[k, nA+j] = -1
            N[k, nA+nB+k] = 1
    # print(N)
    K = K.reshape(-1)

    c = eqtk.solve(c0=c0, N=N, K=K)
    AF, BF, C = c[:nA], c[nA:nA+nB], c[nA+nB:]
    return AF, BF, C




if __name__ == '__main__':

    AT = torch.Tensor([1, 1])
    BT = torch.Tensor([1, 1, 1])
    K  = torch.Tensor([1, 2, 3, 4, 5, 6]).reshape(2,3)
    K.requires_grad = True

    t0 = time.perf_counter()
    for i in range(1000):
        at = AT.numpy()
    t1 = time.perf_counter()
    print('torch to numpy time', t1-t0)

    AT, BT, K = AT.numpy(), BT.numpy(), K.detach().numpy()

    t0 = time.perf_counter()
    for i in range(1000):
        at = torch.from_numpy(AT)
    t1 = time.perf_counter()
    print('numpy to torch time', t1-t0)



    AT = torch.Tensor([1, 1])
    BT = torch.Tensor([1, 1, 1])
    K  = torch.Tensor([1, 2, 3, 4, 5, 6]).reshape(2,3)
    K.requires_grad = True

    t0 = time.perf_counter()
    for i in range(1000):
        with torch.no_grad():
            AF, BF, C = torch_solve(AT, BT, K)
    t1 = time.perf_counter()
    print('torch_solve time', t1-t0)

    t0 = time.perf_counter()
    for i in range(1000):
        with torch.no_grad():
            AF, BF, C = torch_solve(AT, BT, K)
        dC_dK = torch_iterate_last(AT, BT, K, AF, BF, C)
    t1 = time.perf_counter()
    print('torch_solve + torch_iterate_last time', t1-t0)



    AT, BT, K = AT.numpy(), BT.numpy(), K.detach().numpy()

    t0 = time.perf_counter()
    for i in range(1000):
        AF, BF, C = numpy_solve(AT, BT, K)
    t1 = time.perf_counter()
    print('numpy_solve time', t1-t0)

    t0 = time.perf_counter()
    for i in range(1000):
        AF, BF, C = numpy_solve(AT, BT, K)
        dC_dK = numpy_gradient(AF, BF, K)
    t1 = time.perf_counter()
    print('numpy_solve + numpy_gradient time', t1-t0)

    # eqtk first time is much slower
    AF, BF, C = eqtk_solve(AT, BT, K)
    t0 = time.perf_counter()
    for i in range(1000):
        AF, BF, C = eqtk_solve(AT, BT, K)
    t1 = time.perf_counter()
    print('eqtk_solve time', t1-t0)




'''
torch to numpy time 0.0004
numpy to torch time 0.0012
torch_solve time 0.80
torch_solve + torch_iterate_last time 0.86
numpy_solve time 0.20
numpy_solve + numpy_gradient time 0.37
eqtk_solve time 0.20
'''