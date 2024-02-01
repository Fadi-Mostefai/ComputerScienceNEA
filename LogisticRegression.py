import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, lambda_=0.1):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.lambda_ = lambda_  # Regularization parameter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent with L2 Regularization
        for _ in range(self.n_iters):
            model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(model)

            # Regularization term added to the gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + (self.lambda_ / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))





