#!/usr/bin/env python3
import unittest

import numpy as np

from model import ProductionPredictor


class TestProductionPredictor(unittest.TestCase):
    def setUp(self):
        # Generate fake data
        x_base = np.random.uniform(0, 5, (1000,))
        self.x = np.c_[
            x_base,
            x_base + np.random.uniform(-1, 0.5, (1000,)),
            x_base + np.random.uniform(-0.5, 1.0, (1000,))
        ]
        self.y = x_base + np.random.uniform(-1, 1, (1000,))

    def test_error_on_nan(self):
        self.clf = ProductionPredictor()
        self.x[0][0] = np.nan
        with self.assertRaises(AssertionError):
            self.clf.fit(self.x, self.y)

    def test_predict_error_on_not_fitted(self):
        self.clf = ProductionPredictor()
        with self.assertRaises(AssertionError):
            self.clf.predict(self.x)

    def test_score_error_on_not_fitted(self):
        self.clf = ProductionPredictor()
        with self.assertRaises(AssertionError):
            self.clf.score(self.x, self.y)

    def test_fit_error_on_refitting(self):
        self.clf = ProductionPredictor()
        self.clf.fit(self.x, self.y)
        with self.assertRaises(AssertionError):
            self.clf.fit(self.x, self.y)

    def test_fit_no_smoke(self):
        self.clf = ProductionPredictor()
        self.clf.fit(self.x, self.y)

    def test_score(self):
        self.clf = ProductionPredictor()
        self.clf.fit(self.x, self.y)
        y_pred = self.clf.predict(self.x)
        self.assertEqual(y_pred.shape, self.y.shape)

    def test_reasonable_score(self):
        self.clf = ProductionPredictor()
        self.clf.fit(self.x, self.y)
        self.assertLess(self.clf.score(self.x, self.y), 0.5)


if __name__ == "__main__":
    unittest.main()
