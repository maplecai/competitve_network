import time
import numpy as np
import eqtk


nA = 2
nB = 3
nT = nA + nB + nA*nB


if __name__ == '__main__':

    c0 = np.concatenate([np.ones(nA+nB), np.zeros(nA*nB)])
    N = np.zeros((nA*nB, nA + nB + nA*nB))
    for i in range(nA):
        for j in range(nB):
            k = nB*i + j
            N[k, i] = -1
            N[k, nA+j] = -1
            N[k, nA+nB+k] = 1
    # print(N)
    K = np.array([1,2,3,4,5,6])

    c = eqtk.solve(c0=c0, N=N, K=K)

    print(c)

    t0 = time.perf_counter()
    for i in range(1000):
        c = eqtk.solve(c0=c0, N=N, K=K)
    t1 = time.perf_counter()
    print('eqtk solve time', t1-t0)

