# Inputting various modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading the Iris dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
dataset = pd.read_csv(url, names=names)

#Selecting specific rows and columns
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the data into training and testing data into a split 70-30 percentage
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Scaling the Data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Building and training the Logistic Regression Model with the training data
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta):
    return sigmoid(np.dot(X, theta))

def cost_function(X, y, theta):
    m = len(y)
    h = predict(X, theta)
    J = (-1/m) * np.sum((y * np.log(h)) + ((1-y) * np.log(1-h)))
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h = predict(X, theta)
        theta -= alpha * (1/m) * np.dot(X.T, (h - y))
        J_history[i] = cost_function(X, y, theta)
    return theta, J_history

X_train = np.insert(X_train, 0, 1, axis=1)
y_train = np.where(y_train == 'Iris-setosa', 0, np.where(y_train == 'Iris-versicolor', 1, 2))

num_iters = 1000
alpha = 0.1
theta = np.zeros((X_train.shape[1], 1))

theta, J_history = gradient_descent(X_train, y_train.reshape(-1, 1), theta, alpha, num_iters)

#Evaluating the model

X_test = np.insert(X_test, 0, 1)





