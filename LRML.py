import numpy as np


class LinearRegression():
    """THE FOLLOWING METHOD IS THE CONSTRUCTOR METHOD, CONTAINING ALL NECESSARY VARIABLES TO BE USED THROUGHOUT THE LINEAR REGRESSION MODEL"""
    def __init__(self, lr=0.001, n_iterations=1000):
        self.lr = lr  # This is the learning rate
        self.n_iterations = n_iterations  # This is the number of iterations
        self.weights = None  # This is going to be the weights
        self.bias = None  # This is going to be the bias

    """THE FOLLOWING METHOD IS THE FIT METHOD"""
    def fit(self, X, y):
        n_samples, n_features = X.shape()  # Grabs the number of samples and the number of features
        self.weights = np.zeros(n_features)  # We start off setting the weight to 0.
        self.bias = 0  # We start off setting the bias to 0.

        for _ in range(self.n_iterations):  # Run the following code for a set amount of times, that being the number of iterations
            y_pred = np.dot(X, self.weights) + self.bias  # Calculate our prediction variable

            derivative_weights = (1/n_samples) * np.dot(X.T, (y_pred - y))  # Calculate the derivative for the weights
            derivative_bias = (1/n_samples) * np.sum(y_pred - y)  # Calculate the derivative of the bias

            self.weights = self.weights - self.lr*derivative_weights  # Calculating the new weights
            self.bias = self.bias - self.lr*derivative_bias  # Calculating the new bias

    """THE FOLLOWING METHOD IS THE PREDICT METHOD, WHERE WE CALCULATE A ACCURATE PREDICTION"""
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred




