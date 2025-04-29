n, m = map(int, input().split())
s = input()
t = input()
I, D, S = map(int, input().split())

dp = [[0] * (m + 1) for _ in range(n + 1)]

# базовые случаи
for i in range(n + 1):
    dp[i][0] = i * D
for j in range(m + 1):
    dp[0][j] = j * I

# основное заполнение
for i in range(1, n + 1):
    for j in range(1, m + 1):
        if s[i-1] == t[j-1]:
            cost = 0
        else:
            cost = S
        dp[i][j] = min(
            dp[i-1][j] + D,        # удаление
            dp[i][j-1] + I,        # вставка
            dp[i-1][j-1] + cost    # замена / без изменения
        )

print(dp[n][m])
