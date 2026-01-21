# -*- coding: utf-8 -*-
import numpy as np
import math
import pathlib
import matplotlib.pyplot as plt

path = "B.txt"

y = np.array([float(v) for v in pathlib.Path(path).read_text().split()])
N = len(y)
k = np.arange(N)

best = None

for m in range(0, N // 2 + 1):
    omega = 2.0 * math.pi * m / N
    c = np.cos(omega * k)

    denom = float(np.dot(c, c))
    if denom < 1e-12:
        continue

    A = float(np.dot(y, c) / denom)
    xhat = A * c

    mse = float(np.mean((y - xhat) ** 2))
    cand = (mse, m, omega, A, xhat)
    if best is None or cand[0] < best[0]:
        best = cand

mse, m, omega, A, xhat = best

print("N =", N)
print("m =", m)
print("omega =", omega)
print("A =", A)
print("MSE =", mse)

print("omega_4dp =", "{:.4f}".format(omega))
print("A_4dp =", "{:.4f}".format(A))

plt.figure(figsize=(9, 4))
plt.plot(k, y, label="y_k", linewidth=1.0)
plt.plot(k, xhat, label="A*cos(omega*k)", linewidth=1.0)
plt.xlabel("k")
plt.ylabel("value")
plt.title("Noisy data and fitted cosine")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("task2_fit_plot.png", dpi=200)
plt.close()

ms = np.arange(0, N // 2 + 1)
mses = np.zeros_like(ms, dtype=float)

for idx, mm in enumerate(ms):
    ww = 2.0 * math.pi * mm / N
    cc = np.cos(ww * k)
    denom2 = float(np.dot(cc, cc))
    if denom2 < 1e-12:
        mses[idx] = np.nan
        continue
    AA = float(np.dot(y, cc) / denom2)
    mses[idx] = float(np.mean((y - AA * cc) ** 2))

plt.figure(figsize=(9, 4))
plt.plot(ms, mses)
plt.xlabel("m")
plt.ylabel("MSE")
plt.title("MSE vs m for omega = 2*pi*m/N")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("task2_mse_plot.png", dpi=200)
plt.close()


