import numpy as np
from scipy.optimize import curve_fit

data = np.loadtxt('data.csv', delimiter=',')
x_data = data[:, 0]
f_data = data[:, 1]

def model_function(x, a, b, c):
    return ((a * np.sin(x) + b * np.log(x))**2 + c * x**2)

params, _ = curve_fit(model_function, x_data, f_data, bounds=(0, np.inf))

a, b, c = params
print(f"{a:.2f} {b:.2f} {c:.2f}")
