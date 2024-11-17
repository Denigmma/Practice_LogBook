import numpy as np

def hard_sin(x):
    return np.sin(np.log(x ** np.sin(x)))

print(hard_sin(7))