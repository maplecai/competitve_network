import numpy as np
import time
import torch


a = np.array([1.0, 2, 3])
b = np.array([1.0, 2, 3])


t0 = time.perf_counter()
for i in range(1000):
    c = (a * b).sum(axis=0)
t1 = time.perf_counter()
print('np * time', t1-t0)



t0 = time.perf_counter()
for i in range(1000):
    c = np.dot(a, b)
t1 = time.perf_counter()
print('np dot time', t1-t0)



t0 = time.perf_counter()
for i in range(1000):
    c = np.matmul(a, b.reshape(-1,1))
t1 = time.perf_counter()
print('np matmul time', t1-t0)



a = torch.tensor([1.0, 2, 3])
b = torch.tensor([1.0, 2, 3])


t0 = time.perf_counter()
for i in range(1000):
    c = (a * b).sum(axis=0)
t1 = time.perf_counter()
print('torch * + sum time', t1-t0)



t0 = time.perf_counter()
for i in range(1000):
    c = torch.dot(a, b)
t1 = time.perf_counter()
print('torch dot time', t1-t0)



t0 = time.perf_counter()
for i in range(1000):
    c = torch.matmul(a, b.reshape(-1,1))
t1 = time.perf_counter()
print('torch reshape + matmul time', t1-t0)

'''
np * time 0.0023517999943578616
np dot time 0.0012829000042984262
np matmul time 0.0014303000061772764
torch * + sum time 0.005810999995446764
torch dot time 0.0020589000050676987
torch reshape + matmul time 0.006028899995726533
'''