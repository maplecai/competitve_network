import numpy as np
import time
import torch


a = np.array([1.0, 2, 3])
b = torch.tensor([1.0, 2, 3])


aa = np.array([1.0, 2, 3])



t0 = time.perf_counter()
for i in range(1000):
    c = (a * aa).sum(0)
t1 = time.perf_counter()

print(t1-t0)






t0 = time.perf_counter()
for i in range(1000):
    c = np.dot(a, aa)
t1 = time.perf_counter()

print(t1-t0)






t0 = time.perf_counter()
for i in range(1000):
    c = np.matmul(a, aa.reshape(-1,1))
t1 = time.perf_counter()

print(t1-t0)

