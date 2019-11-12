# Time-series solution

See `script.py` for a simple exploration of time-series-based solution. The
problem is predicting the power output in the next half-hour period, given the
real history.

3 approaches:
  - Rolling mean, min max, MAE: 0.1259
  - Autoregressive (regression on previous targets), MAE: 0.561
  - Combined, MAE: 0.553


# Some comments

- Missing values should be interpolated beforehand using before/after values
- While you mention time-series prediction in the introduction, the task is
  said to be predicting a single value from the three vendor's data. This part
  was a bit confusing.
- In the case that we wanted to actually do time-series prediction we could use
  some autoregressive model.


# Solution details

- Focused on a simple solution
- Minimal feature engineering, difference between vendors
- Metric: Mean Absolute Error (MAE) for interpretable results
- Gradient Boosting Regression using [XGBoost](https://en.wikipedia.org/wiki/XGBoost)
- Vendor's predictions MAE: 0.2096, 0.2883, 0.2146
- Our method, 10-fold cross validation MAE: 0.1967.

# Usage

```
from model import ProductionPredictor
clf = ProductionPredictor()
clf.fit(X, y)
# [...]
```

## Testing:

```
./test.py
```
