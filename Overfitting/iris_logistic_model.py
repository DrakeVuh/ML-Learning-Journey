# iris_logistic_model.py

from sklearn.linear_model import LogisticRegression

class IrisLogisticModel:
    def __init__(self, penalty='l2', C=1.0):
        self.model = LogisticRegression(penalty=penalty, C=C, solver='liblinear')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def get_weights(self):
        return self.model.coef_