import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import scipy as sk

if __name__ == "__main__":

    a = pd.read_csv('dataset.csv')
    #X = np.array(a[0])
    #y = np.array(a[1])
    #print(a)

    X = np.array(a['input'], dtype=float)
    y = a['output']


    print(X)
    print(y)