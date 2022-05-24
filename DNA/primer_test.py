import primer3
import numpy as np
from utils import *

class PrimerGenerator:
    def __init__(self):
        self.nt = ['A','G','C','T']
    
    def generate(self, length):
        seq = ''
        for i in range(length):
            j = np.random.randint(0,4)
            seq += self.nt[j]
        return seq


if __name__ == '__main__':
    generator = PrimerGenerator()
    for i in range(10):
        s1 = generator.generate(20)
        s2 = reverse_complement(s1)
        a1 = primer3.calcTm(s1)
        a2 = primer3.calcHeterodimer(s1, s2)
        print(a1)
        print(a2)