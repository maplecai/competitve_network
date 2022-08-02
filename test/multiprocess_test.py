from multiprocessing import Pool
import time
import os

def f(x):
    time.sleep(2)
    return x*x

if __name__ == '__main__':
    # start 4 worker processes
    with Pool(processes=4) as pool:
        res_list = []
        for i in range(10):
            res = pool.apply_async(f, (i,))
            res_list.append(res)
        for i in range(10):
            print(res_list[i].get())


