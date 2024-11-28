### right solution

MOD = 1000000007

def mat_mult(A, B, n):
    #перемножение двух матриц A и B по модулю
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = sum(A[i][p] * B[p][j] for p in range(n)) % MOD
    return C


def mat_pow(A, k, n):
    #возведение матрицы A в степень k по модулю
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]  # ед матр
    base = A

    while k > 0:
        if k % 2 == 1:
            result = mat_mult(result, base, n)
        base = mat_mult(base, base, n)
        k //= 2

    return result

#количество путей длины k из вершины 1 в вершину 1 по модулю
def count_paths(n, k, adj):
    # Сначала возводим матрицу смежности в степень k
    adj_k = mat_pow(adj, k, n)

    # результат adj_k[0][0]так как мы ищем пути из вершины 1 в вершину 1
    return adj_k[0][0]


n, k = map(int, input().split())
adj = [list(map(int, input().split())) for _ in range(n)]

result = count_paths(n, k, adj)

print(result)
