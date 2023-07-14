import numpy as np
import torch
import time
#import eqtk


# def eqtk_solve(AT, BT, K, max_iter=20, tol=1e-4):
#     '''
#     use eqtk to solve the equilibrium
#     '''
#     AT, BT = AT.squeeze(2), BT.squeeze(2)
#     nA, nB = AT.shape[1], BT.shape[1]
    
#     c0 = np.concatenate([AT, BT, np.zeros(shape=(len(AT), nA*nB))], axis=1)
#     N = np.zeros((nA*nB, nA + nB + nA*nB))
#     for i in range(nA):
#         for j in range(nB):
#             k = nB*i + j
#             N[k, i] = -1
#             N[k, nA+j] = -1
#             N[k, nA+nB+k] = 1
#     # print(N)
#     K = K.reshape(-1)

#     c = eqtk.solve(c0=c0, N=N, K=K, tol=tol)
#     AF, BF, C = c[:, :nA], c[:, nA:nA+nB], c[:, nA+nB:]
#     return AF, BF, C



def numpy_solve(AT, BT, K, max_iter=30, tol=1e-6):
    '''
    use numpy to solve the equilibrium
    '''
    AF = np.zeros(AT.shape)
    for iter in range(max_iter):
        AF_0 = AF
        BF = 1 / (K.T @ AF + 1) * BT
        AF = 1 / (K @ BF + 1) * AT
        err = (np.linalg.norm(AF-AF_0, axis=(1,2)) / np.linalg.norm(AT, axis=(1,2))).max()
        #print(AF)
        if (err < tol):
            break
    #print(f'use {iter}/{max_iter} iterations to converge')

    return AF



def torch_solve(AT, BT, K, max_iter=20, tol=1e-3):
    '''
    use torch to solve the equilibrium
    '''
    AF = torch.zeros(AT.shape, dtype=AT.dtype, device=AT.device)
    BF = torch.zeros(BT.shape, dtype=BT.dtype, device=BT.device)
    for iter in range(max_iter+1):
        AF_0 = AF
        BF_0 = BF
        BF = BT / ((K * AF).sum(axis=1, keepdim=True) + 1)
        AF = AT / ((K * BF).sum(axis=2, keepdim=True) + 1)
        #err = torch.abs(AF-AF_0).max() + torch.abs(BF-BF_0).max()
        #err = (torch.linalg.norm(AF-AF_0, axis=(1,2)) / torch.linalg.norm(AT, axis=(1,2))).max()

        err = (torch.abs(BF - BF_0) / BT).max()
        #print(AF)
        if (err < tol):
            break
    #print(f'{iter}/{max_iter} iterations')

    return AF



def one_iter(AT, BT, K, AF):
    BF = BT / ((K * AF).sum(axis=1, keepdim=True) + 1)
    AF = AT / ((K * BF).sum(axis=2, keepdim=True) + 1)
    return AF



def one_iter_grad(AT, BT, K, AF):
    AF_0 = AF
    BF = BT / ((K * AF).sum(axis=1, keepdim=True) + 1)
    AF = AT / ((K * BF).sum(axis=2, keepdim=True) + 1)

    with torch.no_grad():
        dBF = K * BT / (((K * AF_0).sum(axis=1, keepdim=True) + 1) ** 2)
        dAF = K * AT / (((K * BF).sum(axis=2, keepdim=True) + 1) ** 2)
        
        dfun = dAF @ dBF.transpose(2,1)
    return AF, dfun





# AT BT 列向量
def torch_solve_2(AT, BT, K, max_iter=30, tol=1e-6):
    '''
    use torch to solve the equilibrium concentration
    '''
    AF = torch.zeros(AT.shape, dtype=AT.dtype, device=AT.device)
    for iter in range(1, max_iter+1):
        AF_0 = AF
        BF = BT / (K.T @ AF + 1)
        AF = AT / (K @ BF + 1)
        err = (torch.linalg.norm(AF-AF_0, axis=(1,2)) / torch.linalg.norm(AT, axis=(1,2))).max()
        #print(AF)
        if (err < tol):
            break
    #print(f'{iter}/{max_iter} iterations, converge')
    #if (iter == max_iter):
        #print(f'{iter} iterations, still not converge')

    return AF



def one_iter_grad_2(AT, BT, K, AF):
    AF_0 = AF
    BF = 1 / (K.T @ AF + 1) * BT
    AF = 1 / (K @ BF + 1) * AT

    dBF = 1 / ((K.T @ AF_0 + 1) ** 2) * K.T * BT
    dAF = 1 / ((K @ BF + 1) ** 2) * K * AT
    dfun = dAF @ dBF
    return AF, dfun




# # 牛顿法可能迭代出负数，懒得优化了，研究这个不如去用eqtk
# def solve_newton(AT, BT, K, max_iter=20, tol=1e-4):
#     AF = AT.clone().detach()
#     for iter in range(max_iter):
#         BF = 1 / (K.T @ AF + 1) * BT
#         AF_ = 1 / (K @ BF + 1) * AT
#         residual = AF - AF_

#         err = torch.norm(residual) / torch.norm(AT)
#         if (err < tol):
#             break

#         # newton
#         dBF = 1 / ((K.T @ AF + 1) ** 2) * K.T * BT
#         dAF_ = 1 / ((K @ BF + 1) ** 2) * K * AT @ dBF
#         AF = AF - torch.linalg.inv(torch.eye(dAF_.shape[-1]) - dAF_.transpose(1,2)) @ residual
#         print(AF)

#     print(f'use {iter}/{max_iter} iterations to converge')
#     return AF_, dAF_





# def torch_solve_C(AT, BT, K, max_iter=20, tol=1e-3):
#     C = torch.zeros((AT.shape[0], AT.shape[1], BT.shape[2])).to(AT.device)
#     for iter in range(max_iter+1):
#         C_last = C
#         AF = AT - C.sum(axis=2, keepdim=True)
#         BF = BT - C.sum(axis=1, keepdim=True)
#         C = AF * BF * K
#         err = torch.abs(C - C_last).max()
#         if (err < tol):
#             break
#     print(f'{iter}/{max_iter} iterations')
#     return C





# def numpy_gradient(AF, BF, K):
#     '''
#     Using numpy to solve the gradient
#     '''
#     nA = len(AF)
#     nB = len(BF)
#     dC_dK = np.zeros((nA, nB, nA, nB), dtype=np.float32)
#     dA_dK = np.zeros((nA, nA, nB), dtype=np.float32)
#     dB_dK = np.zeros((nB, nA, nB), dtype=np.float32)
    
    
#     # 先求解dAF/dKpq, dBF/dKpq, 解nA+nB维线性方程组
#     W = np.zeros((nA+nB, nA+nB), dtype=np.float32)
#     # 一行一行填W
#     for m in range(nA):
#         # 第m行
#         for j in range(nB):
#             # 第nA+j列
#             W[m, nA+j] += K[m, j] * AF[m]
#             # 第m列
#             W[m, m] += K[m, j] * BF[j]
#         W[m, m] += 1
#     for n in range(nB):
#         # 第nA+n行
#         for i in range(nA):
#             # 第i列
#             W[nA+n, i] += K[i, n] * BF[n]
#             # 第nA+n列
#             W[nA+n, nA+n] += K[i, n] * AF[i]
#         W[nA+n, nA+n] += 1

#     W_inv = np.linalg.inv(W)
    
#     # t0 = time.perf_counter()
#     # for i in range(1000):
#     #     W_inv = np.linalg.inv(W)
#     # t1 = time.perf_counter()
#     # print('process time', t1-t0)

#     # 对于给定的Kpq
#     for p in range(nA):
#         for q in range(nB):
#             b = np.zeros((nA+nB, 1))
#             b[p, 0] += - AF[p] * BF[q]
#             b[nA+q, 0] += - AF[p] * BF[q]
#             x = np.matmul(W_inv, b).reshape(-1)
#             dA_dK[:, p, q] = x[:nA]
#             dB_dK[:, p, q] = x[nA:]

#     # 然后求dC/pKpq
#     for i in range(nA):
#         for j in range(nB):
#             dC_dK[i, j] += K[i, j] * BF[j] * dA_dK[i] + K[i, j] * AF[i] * dB_dK[j]
#             dC_dK[i, j, i, j] += AF[i] * BF[j]

#     return dA_dK, dB_dK, dC_dK




if __name__ == '__main__':

    torch.set_printoptions(precision=16)
    np.set_printoptions(precision=16)
    

    # # tol如何选择，1e-4比较合适
    # AT = torch.FloatTensor([1, 1]).reshape(1,2,1)
    # BT = torch.FloatTensor([1, 1, 1]).reshape(1,3,1)
    # K  = torch.FloatTensor([1, 2, 3, 4, 5, 6]).reshape(2,3)

    # AT = torch.rand(size=(16,2,1))
    # BT = torch.rand(size=(16,3,1))
    # K  = torch.rand(size=(2,3))


    # AF = torch_solve_new(AT, BT, K, tol=1e-3)
    # AF = torch_solve_new(AT, BT, K, tol=1e-4)
    # AF = torch_solve_new(AT, BT, K, tol=1e-5)

    # AF = torch_solve_new(AT * 10, BT, K, tol=1e-3)
    # AF = torch_solve_new(AT * 10, BT, K, tol=1e-4)
    # AF = torch_solve_new(AT * 10, BT, K, tol=1e-5)

    # AF = torch_solve_new(AT, BT * 10, K, tol=1e-3)
    # AF = torch_solve_new(AT, BT * 10, K, tol=1e-4)
    # AF = torch_solve_new(AT, BT * 10, K, tol=1e-5)

    # AF = torch_solve_new(AT, BT, K * 10, tol=1e-3)
    # AF = torch_solve_new(AT, BT, K * 10, tol=1e-4)
    # AF = torch_solve_new(AT, BT, K * 10, tol=1e-5)













    # # 对比不同求解方法的时间

    # # batch
    # AT = torch.rand(size=(16,2,1))
    # BT = torch.rand(size=(16,1,3))
    # K  = torch.rand(size=(2,3))


    # t0 = time.perf_counter()
    # for i in range(1000):
    #     AF = torch_solve(AT, BT, K)
    # t1 = time.perf_counter()
    # print('torch solve time', t1-t0)
    # #print(AF)


    # t0 = time.perf_counter()
    # for i in range(1000):
    #     with torch.no_grad():
    #         AF = torch_solve(AT, BT, K)
    # t1 = time.perf_counter()
    # print('torch solve time', t1-t0)
    # #print(AF)


    # AT = torch.rand(size=(16,2,1))
    # BT = torch.rand(size=(16,3,1))
    # K  = torch.rand(size=(2,3))


    # t0 = time.perf_counter()
    # for i in range(1000):
    #     AF = torch_solve_2(AT, BT, K)
    # t1 = time.perf_counter()
    # print('torch solve time', t1-t0)
    # #print(AF)

    # t0 = time.perf_counter()
    # for i in range(1000):
    #     with torch.no_grad():
    #         AF = torch_solve_2(AT, BT, K)
    # t1 = time.perf_counter()
    # print('torch solve time with no grad', t1-t0)
    # #print(AF)


    # AT, BT, K = AT.numpy(), BT.numpy(), K.numpy()

    # t0 = time.perf_counter()
    # for i in range(1000):
    #     AF = numpy_solve(AT, BT, K)
    # t1 = time.perf_counter()
    # print('numpy solve time', t1-t0)
    # #print(AF)




    # AF, BF, C = eqtk_solve(AT, BT, K)
    # t0 = time.perf_counter()
    # for i in range(1000):
    #     AF, BF, C = eqtk_solve(AT, BT, K)
    # t1 = time.perf_counter()
    # print('eqtk_solve time', t1-t0)
    # print(AF)
    












    # 关于 one_iter_grad 函数梯度是否正确的实验
    torch.manual_seed(0)

    AT = torch.DoubleTensor([1, 1]).reshape(1,2,1) * 2
    BT = torch.DoubleTensor([1, 1, 1]).reshape(1,1,3) * 2
    K  = torch.DoubleTensor([1, 2, 3, 4, 5, 6]).reshape(2,3).requires_grad_(True)

    AT = torch.rand(size=(16,2,1))
    BT = torch.rand(size=(16,1,3))
    K  = torch.rand(size=(2,3))

    AF = torch_solve(AT, BT, K)
    print(AF)
    j = torch.autograd.functional.jacobian(func=lambda x: one_iter(AT, BT, K, x), inputs=AF)[:, :, 0, :, :, 0].diagonal(dim1=0, dim2=2).permute(2,0,1)
    print('autograd jacobian', j.shape, j)


    AF = torch_solve(AT, BT, K)
    print(AF)
    AF_1, dfunc_1 = one_iter_grad(AT, BT, K, AF)
    print('dfunc', dfunc_1)


    AF = torch_solve_2(AT, BT.transpose(2,1), K)
    print(AF)
    AF_1, dfunc_1 = one_iter_grad_2(AT, BT.transpose(2,1), K, AF)
    print('dfunc', dfunc_1)

    for delta in [1e-3, 1e-4, 1e-5]:

        dAF = torch.DoubleTensor([1, 0]).reshape(1,2,1) * delta
        AF_2, dfunc_2 = one_iter_grad(AT, BT, K, AF+dAF)
        gradient_0 = (AF_2-AF_1)/delta

        dAF = torch.DoubleTensor([0, 1]).reshape(1,2,1) * delta
        AF_2, dfunc_2 = one_iter_grad(AT, BT, K, AF+dAF)
        gradient_1 = (AF_2-AF_1)/delta

        gradient = torch.cat([gradient_0, gradient_1], dim=2)
        print('numerical dAF/dAF', delta, gradient)








'''

torch to numpy time 0.00047082453966140747
torch_solve time 0.9984057322144508
numpy to torch time 0.0009172633290290833
numpy_solve time 0.22983410209417343

'''
