import numpy as np, math, pathlib
import numpy.linalg as la
import matplotlib.pyplot as plt

path = r"5be57622-a4ad-a5e4026cc6fb.txt"

x = np.array([float(v) for v in pathlib.Path(path).read_text().split()])
L = len(x)

def detect_period(x, tol=1e-8):
    L = len(x)
    for p in range(1, L + 1):
        if L % p != 0:
            continue
        ok = True
        for i in range(0, L - p, p):
            if np.max(np.abs(x[i:i+p] - x[i+p:i+2*p])) > tol:
                ok = False
                break
        if ok:
            return p
    return L

N = detect_period(x)
xN = x[:N]
k = np.arange(N)

m_min = int(math.ceil(1.0 * N))
m_max = int(math.floor(1.5 * N))

best = None
for m in range(m_min, m_max + 1):
    omega = 2 * math.pi * m / N
    M = np.column_stack([np.sin(omega * k), np.cos(omega * k)])

    B, C = la.lstsq(M, xN, rcond=None)[0]
    xhat = M @ np.array([B, C])

    rmse = la.norm(xhat - xN) / math.sqrt(N)
    A = math.hypot(B, C)
    phi = math.atan2(C, B)

    cand = (rmse, m, omega, A, phi, B, C, xhat)
    if best is None or cand[0] < best[0]:
        best = cand

rmse, m, omega, A, phi, B, C, xhat = best

print("N =", N)
print("m =", m)
print("omega =", omega)
print("A =", A)
print("phi =", phi)
print("RMSE =", rmse)

print("omega_4dp =", "{:.4f}".format(omega))
print("A_4dp =", "{:.4f}".format(A))
print("phi_4dp =", "{:.4f}".format(phi))

plt.figure(figsize=(8, 4))
plt.plot(k, xN, marker='o', linestyle='-', label='data x_k')
plt.plot(k, xhat, marker='x', linestyle='--', label='fit A*sin(omega*k+phi)')
plt.xlabel("k")
plt.ylabel("x_k")
plt.title("N={}, m={}, omega={:.6f}, A={:.6f}, phi={:.6f}".format(N, m, omega, A, phi))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("fit_plot.png", dpi=200)
plt.close()