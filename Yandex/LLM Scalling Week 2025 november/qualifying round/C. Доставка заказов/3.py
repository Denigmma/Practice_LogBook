import sys

INF = 10**9

def manhattan_transform(A, H, D, n, m):
    for i in range(n):
        row = A[i]
        hr = H[i]
        best = INF

        for j in range(m):
            v = row[j]
            b1 = best + 1
            if b1 < v:
                v = b1
            hr[j] = v
            best = v

        best = INF
        for j in range(m - 1, -1, -1):
            v = hr[j]
            b1 = best + 1
            if b1 < v:
                v = b1
                hr[j] = v
            best = v

    for j in range(m):
        best = INF
        # top -> bottom
        for i in range(n):
            v = H[i][j]
            b1 = best + 1
            if b1 < v:
                v = b1
            D[i][j] = v
            best = v
        best = INF
        for i in range(n - 1, -1, -1):
            v = D[i][j]
            b1 = best + 1
            if b1 < v:
                v = b1
                D[i][j] = v
            best = v

def solve():
    data = sys.stdin.read().strip().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    sx = int(next(it)) - 1
    sy = int(next(it)) - 1

    grid = []
    pos = [[] for _ in range(26)]
    for i in range(n):
        row = list(next(it).strip())
        grid.append(row)
        for j, ch in enumerate(row):
            pos[ord(ch) - 97].append((i, j))

    s = list(next(it).strip())

    s_comp = []
    prev = None
    for ch in s:
        if ch != prev:
            s_comp.append(ch)
            prev = ch

    A = [[INF]*m for _ in range(n)]
    A[sx][sy] = 0

    H = [[INF]*m for _ in range(n)]
    D = [[INF]*m for _ in range(n)]

    last_set = [(sx, sy)]

    ans = 0
    for step, ch in enumerate(s_comp):
        manhattan_transform(A, H, D, n, m)

        for (i, j) in last_set:
            A[i][j] = INF

        tgt = ord(ch) - 97
        last_set = pos[tgt]

        if step == len(s_comp) - 1:
            best = INF
            for (i, j) in last_set:
                v = D[i][j]
                A[i][j] = v
                if v < best:
                    best = v
            ans = best
        else:
            for (i, j) in last_set:
                A[i][j] = D[i][j]

    print(ans)

if __name__ == "__main__":
    solve()
