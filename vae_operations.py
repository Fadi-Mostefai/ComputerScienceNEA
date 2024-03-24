import numpy as np


# Contains static methods for various operations commonly used in Variational Autoencoders (VAE)
class VaeOperations:
    # Computes the sigmoid activation function for an input array 'x'.
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    #  Calculates the derivative of the sigmoid function with respect to its input 'x'.
    @staticmethod
    def sigmoid_derivative(x):
        return VaeOperations.sigmoid(x) * (1 - VaeOperations.sigmoid(x))

    # Implements the Rectified Linear Unit (ReLU) activation function
    # defined as the positive part of its argument: max(0,x).
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    # Derivative of the ReLU function, which is 1 for x>0 and 0 otherwise.
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    # Converts an input array x into a probability distribution over predicted output classes
    # by taking the exponential of each element and normalizing these values
    # by dividing by the sum of all the exponentials.
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # Computes the Mean Squared Error (MSE) loss between the true labels 'y_true' and the predicted labels 'y_pred'.
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    # Calculates the derivative of the MSE loss function with respect to the predicted labels 'y_pred'.
    @staticmethod
    def mean_squared_error_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

    # Calculates the Cross-Entropy loss for binary classification tasks,
    # which quantifies the difference between two probability distributions -
    # the true labels and the predicted probabilities.
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Provides the gradient of the Cross-Entropy loss with respect to the predictions 'y_pred'.
    @staticmethod
    def cross_entropy_loss_derivative(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)

    # Applies L2 regularization to the weight parameters, penalizing large values in the weights to prevent overfitting.
    @staticmethod
    def l2_regularization(weights, lambda_param):
        reg_loss = 0.0
        for weight_matrix in weights.values():
            reg_loss += np.sum(np.square(weight_matrix))
        return 0.5 * lambda_param * reg_loss

    # Computes the derivative of the L2 regularization term with respect to the weights,
    @staticmethod
    def l2_regularization_derivative(weights, lambda_param):
        gradients = {}
        for key, weight_matrix in weights.items():
            gradients[key] = lambda_param * weight_matrix
        return gradients