
Experiment 3 — Ordinary Least Squares (Closed-form) Linear Regression


AIM:

To implement Ordinary Least Squares (OLS) linear regression via normal equations and compare the results with Scikit-Learn's LinearRegression.

ALGORITHM:

1. Start and import numpy, LinearRegression, and mean_squared_error.
2. Prepare dataset: Define feature matrix X ([size, age]) and target vector y (price).
3. Add intercept column: Prepend a column of ones to X using np.hstack to form the design matrix X0.
4. Compute normal equation: Calculate the closed-form parameter vector using the formula \(\hat{\beta} = (X_0^T X_0)^{-1} X_0^T y\).
5. Generate closed-form predictions: Multiply X0 by beta_hat to obtain y_pred_closed.
6. Fit Scikit-Learn model: Train LinearRegression().fit(X, y) and predict y_pred_sk for comparison.
7. Evaluate and compare: Print the rounded coefficients beta_hat and compare the Mean Squared Error (mean_squared_error) of both methods.
8. Stop.

THEORY:

Ordinary Least Squares (OLS) is a statistical method for estimating unknown parameters in a linear regression model by minimizing the sum of squared residuals between observed and predicted target values. When the Gram matrix (\(X^{T}X\)) is full-rank and invertible, OLS has an analytical closed-form solution given by the normal equation: \(\hat{\beta} = (X^T X)^{-1} X^T y\). In practice, numerical libraries like Scikit-Learn use Singular Value Decomposition (SVD) rather than explicit matrix inversion to ensure numerical stability when features are collinear or ill-conditioned.

PROGRAM:

python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Toy dataset: features = [size, age], target = price
X = np.array([,,,,])
y = np.array()

# Add intercept term
X0 = np.hstack([np.ones((X.shape, 1)), X])
beta_hat = np.linalg.inv(X0.T.dot(X0)).dot(X0.T).dot(y)
y_pred_closed = X0.dot(beta_hat)

# sklearn model (for comparison)
model = LinearRegression().fit(X, y)
y_pred_sk = model.predict(X)

print('Closed-form beta:', np.round(beta_hat, 3))
print('MSE (closed):', mean_squared_error(y, y_pred_closed))
print('MSE (sklearn):', mean_squared_error(y, y_pred_sk))
Use code with caution.

DETAILED EXPLANATION:

1. Prepare predictors and target: Defines a 5-sample toy dataset X with two features (size and age) and target vector y (price).
2. Add intercept column: X0 = np.hstack([np.ones((X.shape[0], 1)), X]) prepends a column of 1s so the model can learn the bias/intercept term \(\beta _{0}\).
3. Closed-form calculation: np.linalg.inv(X0.T.dot(X0)).dot(X0.T).dot(y) directly implements the normal equation \(\hat{\beta} = (X_0^T X_0)^{-1} X_0^T y\).
4. Numerical stability note: While explicit matrix inversion (np.linalg.inv) is shown for clarity, np.linalg.solve or SVD is preferred in practice to avoid precision loss.
5. Scikit-Learn comparison: LinearRegression().fit(X, y) computes coefficients using SVD-based least squares.
6. Compare MSE: mean_squared_error verifies that predictions from the closed-form normal equation and Scikit-Learn match identically.

SAMPLE OUTPUT:

text
Closed-form beta: [10.5  2.   1.8]
MSE (closed): 0.0
MSE (sklearn): 0.0
Use code with caution.

RESULT:

Thus, closed-form Ordinary Least Squares regression was implemented via the normal equation and compared against Scikit-Learn's LinearRegression, confirming matching coefficients and error metrics while highlighting the importance of SVD for numerical stability in practice.
