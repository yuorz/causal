import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class CausalLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        c: np.ndarray,
        lam: float = 1.0,
        lr: float = 1e-4,        # 先用小一點
        max_iter: int = 50000,
    ):
        self.c = np.asarray(c, float)
        self.lam = lam
        self.lr = lr
        self.max_iter = max_iter

    def _sigmoid(self, z):
        # 稍微穩定一點的寫法
        z = np.asarray(z, float)
        # clip 避免 exp 溢出
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)

        n, d = X.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0

        # c: scalar or (d,)
        c = np.asarray(self.c, float)
        if c.ndim == 0:
            c = np.full(d, float(c))
        elif c.shape[0] != d:
            raise ValueError(f"c must be scalar or shape ({d},), got {c.shape}")
        self.c_ = c

        for _ in range(self.max_iter):
            logits = X @ self.w_ + self.b_
            p = self._sigmoid(logits)

            grad_w = X.T @ (p - y) / n
            grad_b = np.mean(p - y)

            w_tmp = self.w_ - self.lr * grad_w
            self.b_ -= self.lr * grad_b

            thresh = self.lr * self.lam * self.c_
            self.w_ = np.sign(w_tmp) * np.maximum(np.abs(w_tmp) - thresh, 0.0)

            if not np.all(np.isfinite(self.w_)):
                print("Warning: non-finite weights encountered, stopping early")
                break

        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
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
        lr: float = 1e-4,
        max_iter: int = 50000,
        tol: float = 1e-6,        # 收斂條件
        verbose: bool = False,    # optional
    ):
        self.c = np.asarray(c, float)
        self.lam = lam
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)

        # 標準化 X
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1.0
        X = (X - X_mean) / X_std
        self._X_mean = X_mean
        self._X_std = X_std

        # ⭐ 也標準化 y
        y_mean = y.mean()
        y_std = y.std()
        if y_std == 0:
            y_std = 1.0
        y_scaled = (y - y_mean) / y_std
        self._y_mean = y_mean
        self._y_std = y_std

        n, d = X.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0

        if self.c.shape[0] != d:
            raise ValueError(f"c must have shape ({d},), got {self.c.shape}")
        c = self.c

        prev_w = self.w_.copy()

        for it in range(self.max_iter):
            pred = X @ self.w_ + self.b_
            grad_w = (X.T @ (pred - y_scaled)) / n
            grad_b = np.mean(pred - y_scaled)

            w_tmp = self.w_ - self.lr * grad_w
            self.b_ -= self.lr * grad_b

            thresh = self.lr * self.lam * c
            self.w_ = np.sign(w_tmp) * np.maximum(np.abs(w_tmp) - thresh, 0.0)

            max_change = np.max(np.abs(self.w_ - prev_w))

            if max_change < self.tol:
                if self.verbose:
                    print(f"Converged at iter {it}, Δw={max_change:.2e}")
                break
            
            if it % 1000 == 0 and self.verbose:
                mse = np.mean((pred - y_scaled) ** 2)
                print(f"iter {it}, max_change={max_change:.2e}, mse={mse:.4f}")
            
            prev_w = self.w_.copy()

        return self

    def predict(self, X):
        X = np.asarray(X, float)
        X = (X - self._X_mean) / self._X_std
        y_scaled_pred = X @ self.w_ + self.b_
        # 把標準化的 y scale 回原本單位
        return y_scaled_pred * self._y_std + self._y_mean


