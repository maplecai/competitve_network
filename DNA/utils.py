import numpy as np

# 反向
def reverse(seq):
    return seq[::-1]

# 互补
def complement(seq):
    dic = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
    seq1 = ''
    for c in seq:
        seq1 += dic[c]
    return seq1

# 反向互补
def reverse_complement(seq):
    return reverse(complement(seq))

def seq2onthot(seq):
    dic = {'A':0, 'C':1, 'G':2, 'T':3}
    onehot = np.zeros((4, len(seq)))
    for i, c in enumerate(seq):
        onehot[dic[c], i] = 1
    return onehot

def onehot2seq(onehot):
    num = np.argmax(onehot, axis=0)
    seq = num2seq(num)
    return seq

def num2seq(num):
    dic = {0:'A', 1:'C', 2:'G', 3:'T'}
    seq = ''
    for i in num:
        seq += dic[i]
    return seq
