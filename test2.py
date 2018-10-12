import pandas as pd
import numpy as np


if __name__ == "__main__":

    a = pd.DataFrame(pd.read_csv('dataset.csv', header=None))
    X = np.array(a[0])
    y = np.array(a[1])

    syn0 = 2 * np.random.random((3, 4)) - 1
    syn1 = 2 * np.random.random((4, 1)) - 1
    for j in range(60000):
        l1 = 1 / (1 + np.exp(-(np.dot(X, syn0))))
        l2 = 1 / (1 + np.exp(-(np.dot(l1, syn1))))
        l2_delta = (y - l2) * (l2 * (1 - l2))
        l1_delta = l2_delta.dot(syn1.T) * (l1 * (1 - l1))
        syn1 += l1.T.dot(l2_delta)
        syn0 += X.T.dot(l1_delta)