# vae_operations.py

import numpy as np

class VaeOperations:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    @staticmethod
    def mean_squared_error_derivative(y_true, y_pred):
        return y_pred - y_true

    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def cross_entropy_loss_derivative(y_true, y_pred):
        return y_pred - y_true

    @staticmethod
    def l2_regularization(weights, lambda_param):
        return 0.5 * lambda_param * np.sum(np.square(weights))

    @staticmethod
    def l2_regularization_derivative(weights, lambda_param):
        return lambda_param * weights
