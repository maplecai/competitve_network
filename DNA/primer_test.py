import primer3
import numpy as np
from DNAutils import *

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
    dg_list = []
    for i in range(3):
        s1 = generator.generate(20)
        # a1 = primer3.calcTm(s1, tm_method='santalucia')
        # print(a1)
        s2 = reverse_complement(s1)
        s2 = generator.generate(20)
        a2 = primer3.calcHeterodimer(s1, s2, mv_conc=50, dv_conc=0, dntp_conc=0, temp_c=25, dna_conc=1)
        
        print(a2)


        t1 = a2.dh / a2.ds
        print('在K=1时 T=', t1-273)

        g = a2.dh - a2.ds*(25+273)
        K = np.exp(-g / (1.987*(25+273)))
        print('在T=25时 K=', K)
        
        g = a2.dh - a2.ds*(a2.tm+273)
        K = np.exp(-g / (1.987*(a2.tm+273)))
        print('在T=Tm时 K=', K)
        # dna_conc指总单链浓度，GHS单位都是cal

