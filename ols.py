import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Toy dataset: features = [size, age], target = price
X = np.array([
    [50, 10],
    [80, 5],
    [60, 8],
    [100, 2],
    [45, 12]
])

y = np.array([150, 240, 180, 320, 140])

# Add intercept term
X0 = np.hstack([np.ones((X.shape[0], 1)), X])

beta_hat = np.linalg.inv(X0.T.dot(X0)).dot(X0.T).dot(y)

y_pred_closed = X0.dot(beta_hat)

# sklearn model (for comparison)
model = LinearRegression().fit(X, y)
y_pred_sk = model.predict(X)

print("Closed-form beta:", np.round(beta_hat, 3))
print("MSE (closed):", mean_squared_error(y, y_pred_closed))
print("MSE (sklearn):", mean_squared_error(y, y_pred_sk))
