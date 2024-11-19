import numpy as np

N, k, m = map(int, input().split())
X = np.array([list(map(float, input().split())) for _ in range(N)])

predictions = np.random.rand(N)

for iteration in range(m):
    print(" ".join(map(str, predictions)))

    if iteration < m - 1:
        gradients = np.random.rand(N)
        predictions -= 0.1 * gradients

    import sys

    sys.stdout.flush()
