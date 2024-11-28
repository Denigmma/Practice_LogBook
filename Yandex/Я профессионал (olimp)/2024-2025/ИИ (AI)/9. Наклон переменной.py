### no right solution (insufficient accuracy of the response)

# 2
# 0.6732252240180969 0.7077440023422241
# 0.9007378220558167 0.8930410742759705
# 0.855198323726654 0.2551969289779663
# 0.1707950283190436 -0.1707950283190436

import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


def compute_dL_dZ(z, V):
    m = len(z)
    softmax_z = softmax(z)

    #матрица производной softmax
    softmax_derivative = np.diag(softmax_z) - np.outer(softmax_z, softmax_z)

    dL_dZ = softmax_derivative @ V @ np.ones(m)

    return dL_dZ


m = int(input())
z = list(map(float, input().split()))
V = np.array([list(map(float, input().split())) for _ in range(m)])

dL_dZ = compute_dL_dZ(z, V)

print(" ".join(f"{x:.16f}" for x in dL_dZ))
