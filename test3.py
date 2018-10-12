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

    X = a.drop('output', axis=1)
    y = a
    print(X)
    print(y)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=5)

    mlp.fit(X_train, y_train)

    predictions = mlp.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    len(mlp.coefs_)

    len(mlp.coefs_[0])

    len(mlp.intercepts_[0])
    #print(X)
    #print(y)