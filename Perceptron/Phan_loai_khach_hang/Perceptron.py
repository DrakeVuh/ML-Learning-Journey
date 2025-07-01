import numpy as np

class Perceptron:
    def __init__(self, n_features, learning_rate=1.0, max_iter=1000):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = np.zeros(n_features)
        self.bias = 0

    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias >= 0, 1, -1)
    
    def fit(self, X, y):
        for _ in range(self.max_iter):
            predictions = self.predict(X)
            errors = y - predictions
            self.weights += self.learning_rate * np.dot(X.T, errors)
            self.bias += self.learning_rate * np.sum(errors)

    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias >= 0, 1, -1)