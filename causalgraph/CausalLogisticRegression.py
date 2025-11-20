import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class CausalLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    用 weighted L1 做 logistic regression:
        loss = logistic_loss + lam * sum_i c_i |w_i|
    (二元分類版本)
    """
    def __init__(
        self,
        c: np.ndarray,
        lam: float = 1.0,
        lr: float = 1e-3,
        max_iter: int = 2000,
    ):
        self.c = c
        self.lam = lam
        self.lr = lr
        self.max_iter = max_iter

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)

        n, d = X.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        c = self.c

        for _ in range(self.max_iter):
            logits = X @ self.w_ + self.b_
            p = self._sigmoid(logits)

            # gradient
            grad_w = X.T @ (p - y) / n
            grad_b = np.mean(p - y)

            # step
            w_tmp = self.w_ - self.lr * grad_w
            self.b_ -= self.lr * grad_b

            # proximal L1 with weights c_i
            thresh = self.lr * self.lam * c
            self.w_ = np.sign(w_tmp) * np.maximum(np.abs(w_tmp) - thresh, 0.0)

        return self

    def predict_proba(self, X):
        logits = X @ self.w_ + self.b_
        p = self._sigmoid(logits)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class CausalLinearRegression(BaseEstimator, RegressorMixin):
    """
    Regression with feature-wise weighted L1 penalty:
        loss = MSE + lam * sum_i c_i |w_i|
    """
    def __init__(
        self,
        c: np.ndarray,
        lam: float = 1.0,
        lr: float = 1e-3,
        max_iter: int = 2000,
    ):
        self.c = np.asarray(c, float)
        self.lam = lam
        self.lr = lr
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)

        n, d = X.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        c = self.c

        for _ in range(self.max_iter):
            # prediction
            pred = X @ self.w_ + self.b_

            # gradient of MSE
            grad_w = (X.T @ (pred - y)) / n
            grad_b = np.mean(pred - y)

            # gradient step
            w_tmp = self.w_ - self.lr * grad_w
            self.b_ -= self.lr * grad_b

            # weighted L1 proximal step
            thresh = self.lr * self.lam * c
            self.w_ = np.sign(w_tmp) * np.maximum(np.abs(w_tmp) - thresh, 0.0)

        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return X @ self.w_ + self.b_