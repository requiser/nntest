import numpy as np
import pandas as pd
import scipy as sp

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = 1
        self.weights2   = 1
        self.y          = y
        self.output     = self.input

    def feedforward(self):
    #    self.layer1 = sigmoid(np.dot(self.input, self.weights1))
    #    self.output = sigmoid(np.dot(self.layer1, self.weights2))
        self.layer1 = np.dot(self.input, self.weights1)
        self.output = np.dot(self.layer1, self.weights2)

if __name__ == "__main__":

    a = pd.read_csv('dataset.csv')
    X = np.array(a["input"])
    y = np.array(a["output"])
    nn = NeuralNetwork(X, y)
    print(X)
    for i in range(1500):
        nn.feedforward()

    print(nn.output)