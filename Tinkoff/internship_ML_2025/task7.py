MOD = 998244353
inv2 = (MOD + 1) // 2
n, k = map(int, input().split())
a = list(map(int, input().split()))

A = [0] * (k + 1)
A[0] = n
for x in a:
    val = 1
    for m in range(1, k + 1):
        val = (val * x) % MOD
        A[m] = (A[m] + val) % MOD
comb = [[0] * (k + 1) for _ in range(k + 1)]
for p in range(k + 1):
    comb[p][0] = 1
    for m in range(1, p + 1):
        comb[p][m] = (comb[p - 1][m] + comb[p - 1][m - 1]) % MOD

pow2 = [1] * (k + 1)
for p in range(1, k + 1):
    pow2[p] = (pow2[p - 1] * 2) % MOD

for p in range(1, k + 1):
    total = 0
    for m in range(p + 1):
        total += comb[p][m] * A[m] % MOD * A[p - m] % MOD
        total %= MOD
    tmp = (A[p] * pow2[p]) % MOD
    total = (total - tmp) % MOD
    ans = (total * inv2) % MOD
    print(ans)