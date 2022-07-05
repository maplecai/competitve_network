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
print('torch * time', t1-t0)



t0 = time.perf_counter()
for i in range(1000):
    c = torch.dot(a, b)
t1 = time.perf_counter()
print('torch dot time', t1-t0)



t0 = time.perf_counter()
for i in range(1000):
    c = torch.matmul(a, b.reshape(-1,1))
t1 = time.perf_counter()
print('torch matmul time', t1-t0)

'''
np * time 0.002249400000437163
np dot time 0.0013595000054920092
np matmul time 0.0013430000108201057
torch * time 0.00600640001357533
torch dot time 0.002086000007693656
torch matmul time 0.005398399996920489
'''