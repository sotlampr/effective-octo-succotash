import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


class ProductionPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.clf = None

    @staticmethod
    def _append_vendor_differences(x):
        """Compute differences between the three vendors for each row
        and append these as 3 new columns.
        Args:
            x (np.ndarray, shape=(?, 3)): the observations
        Returns:
            np.ndarray, shape=(?, 6)
        """
        return np.c_[
            x,
            x[:, 1] - x[:, 0],
            x[:, 2] - x[:, 0],
            x[:, 2] - x[:, 1]
        ]

    def fit(self, X, y, **fit_params):
        assert not np.isnan(X).any(), "Input contains NaN"
        assert self.clf is None, "Estimator has been already fitted."
        X = self._append_vendor_differences(X)
        clf = XGBRegressor(objective="reg:squarederror", n_jobs=-1)
        self.clf = clf.fit(X, y)
        return self

    def predict(self, X):
        assert self.clf is not None
        X = self._append_vendor_differences(X)
        return self.clf.predict(X)

    def score(self, X, y):
        assert self.clf is not None, "Estimator has not been fitted yet."
        X = self._append_vendor_differences(X)
        y_pred = self.clf.predict(X)
        return mean_absolute_error(y, y_pred)
