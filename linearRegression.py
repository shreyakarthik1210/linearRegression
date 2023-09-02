import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math


def main():
    #Creates a random data set and graphs it
    X, y = make_regression(n_samples= 1000, n_features= 1, n_targets= 1, noise=10)
    plt.scatter(X, y, marker = 'x', label = 'y')
    plt.xlabel('X(Features)')
    plt.ylabel('Y(Target)')
    plt.show()
    #Creates linear regression of data above
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, train_size= 0.80, random_state=1)
    print("Shape of X_train " + str(X_train.shape))
    print("Shape of X_test " + str(X_test.shape))
    print("Shape of y_train " + str(y_train.shape))
    print("Shape of y_test " + str(y_test.shape))
    Lmodel = LinearRegression()
    model = Lmodel.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    print(math.sqrt(mean_squared_error(y_train, pred_train)))
    pred_test = model.predict(X_test)
    print(math.sqrt(mean_squared_error(y_test, pred_test)))
    plt.scatter(X_test, y_test, marker = 'o', label = 'y')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
    #Creates a line of best fit
    plt.scatter(X_test, y_test, marker = 'o', label = 'y')
    plt.plot(X_test, pred_test, color= 'Red', linewidth = 3)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

main()
