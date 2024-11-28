### answer=1/1575 or 0,00063492

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n_samples = 10000000
X = np.random.uniform(0, 1, n_samples)

Y_true = X**3
Y1,Y2 = X,X**2

features = np.vstack([X, X**2]).T

model = LinearRegression(fit_intercept=False)

model.fit(features, Y_true)
Y_pred = model.predict(features)

mse = mean_squared_error(Y_true, Y_pred)
print(" MSE:",mse)