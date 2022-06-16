import numpy as np

'''
AT = np.array([1., 1.]).reshape(2,1)
BT = np.array([1., 1.]).reshape(1,2)
K = np.array([1., 2., 3., 4.]).reshape(2,2)

AF = np.array([0.5, 0.5]).reshape(2,1)
BF = np.array([0.5, 0.5]).reshape(1,2)

print(K*BF)
print((K * BF).sum(axis=1, keepdims=True))

AF = AT / ((K * BF).sum(axis=1, keepdims=True) + 1)
BF = BT / ((K * AF).sum(axis=0, keepdims=True) + 1)

print(AF)
print(BF)

C = K*AF*BF
print(C)
'''



import torch
K = torch.tensor([1.])

print(K.shape)
K.reshape(-1)
print(K.shape)

print(K[0].shape)
