import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, lambda_param=0.1):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iters):
            model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(model)

            # Partial derivatives
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y)) + (self.lambda_param / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(model)
        return predictions

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in probabilities]

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))





