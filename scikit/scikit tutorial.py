import matplotlib
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["figure.figsize"] = 10, 5
#%matplotlib inline

# load the iris dataset from sklearn
iris = datasets.load_iris()

# separate features and targets
X = iris.data
y = iris.target

# now we'lll use 'train_test_split' from sklearn
# to sli the data into training and testing sets
test_size = 0.3 # could also specify 'train_size = 0.7' instead
random_state = 0

#
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = test_size,
                                                    random_state = random_state
                                                    )
#
sc = StandardScaler()

#
sc.fit(X_train)

#
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#
print("unique labels: {0}".format(np.unique(y)))

#
n_iter = 40
eta0 = 0.1

#
ppn = Perceptron(n_iter = n_iter,
                 eta0 = eta0,
                 random_state = random_state
                 )

#
ppn.fit(X_train_std,
        y_train)

#
y_pred = ppn.predict(X_train_std)

#
print("accuracy1: {0:.2f}".
      format(accuracy_score(y_test,
                            y_pred) * 100))
