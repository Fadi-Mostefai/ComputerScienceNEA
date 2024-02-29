import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, lambda_param=0.1):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        # Weights and Bias will be set during the fitting process.
        self.weights = None
        self.bias = None

    # The following method fits the logistic regression model to the training data 'X' (features) and 'y' (target labels).
    def fit(self, X, y):
        # Initializes the weights as a zero vector of shape 'n_features' and the bias as 0.
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent + updating the weights and bias after each iteration.
        for _ in range(self.n_iters):
            model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(model)

            # Partial derivatives
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y)) + (self.lambda_param / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # The following method calculates the probability that each instance in 'X' belongs to the positive class (1).
    def predict_proba(self, X):
        # Computes the model's linear combination
        model = np.dot(X, self.weights) + self.bias
        # Applies the sigmoid function to squash the output into the range [0, 1], representing probabilities.
        predictions = self._sigmoid(model)
        return predictions

    # The following method predicts the class labels for the instances in X, based on a specified threshold (default is 0.5).
    def predict(self, X, threshold=0.5):
        # Classifies each instance as 1 (positive class) if its predicted probability is greater than the threshold, else as 0 (negative class).
        probabilities = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in probabilities]

    # A private method that implements the sigmoid function, used to convert the linear combination of inputs into probabilities.
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))





