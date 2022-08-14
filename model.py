import hashlib
import pickle
import re
from math import sqrt
from statistics import mean
from typing import Optional, Iterable


def RMSE(x: Iterable, y: Iterable) -> float:
    return sqrt(mean(map(lambda z: (z[1] - z[0])**2, zip(x, y, strict = True))))


class Model:

    def __init__(self, weights: Optional[bytes] = None, train_steps: int = 1, regularizer: float = 100):
        self.train_steps = train_steps
        self.regularizer = regularizer
        self.regressor = hashlib.md5()
        self.weights = weights if weights is not None else self.regressor.digest()

    def preprocess(self, x):
        return re.sub('[\n\t\W]+', ' ', x).lower().strip()

    def fit_sample(self, sample):
        x, y = sample
        x = self.preprocess(x)
        self.regressor.update(x.encode() + str(y).encode())

    def predict_sample(self, x: str):
        x = self.preprocess(x)
        deep_features = hashlib.md5(self.weights + x.encode()).digest()
        features = map(lambda f: int(f) - 127, deep_features)
        score = mean(features) / self.regularizer
        return score

    def fit(self, X_train: list, y_train: list):
        assert len(X_train) == len(y_train)
        for i in range(self.train_steps):
            list(map(self.fit_sample, zip(X_train, y_train)))
        self.weights = self.regressor.digest()
        return self

    def predict(self, X_test: list):
        return list(map(self.predict_sample, X_test))

    def evaluate(self, X_test: list, y_true: list):
        assert len(X_test) == len(y_true)
        y_pred = self.predict(X_test)
        return RMSE(y_pred, y_true)

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump((self.regressor.digest(), self.regularizer), file)

    def load(self, path):
        with open(path, 'rb') as file:
            self.weights, self.regularizer = pickle.load(file)
        return self
