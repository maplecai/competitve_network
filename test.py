import numpy as np

AT = np.array([1,2]).reshape(2,1)
BT = np.array([3,4]).reshape(1,2)
K = np.array([1,2,3,4]).reshape(2,2)
AF = np.zeros(AT.shape)
BF = np.zeros(BT.shape)


AF = AT / ((K * BF).sum(axis=1, keepdims=True) + 1)
BF = BT / ((K * BF).sum(axis=0, keepdims=True) + 1)

print(AF.shape)
print(BF.shape)
