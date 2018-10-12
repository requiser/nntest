import sklearn as skl
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier

a = pd.read_csv('dataset.csv')
y = a.drop('input',axis=1)
X = a.drop('output',axis=1)
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(X, y)