MOD = 10 ** 9 + 7


def count_valid_sequences(n, sequence):
    # DP таблица: dp[i][j] хранит количество способов до позиции i с j открытыми скобками
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    # Начальное условие: 0 позиций и 0 открытых скобок
    dp[0][0] = 1

    # Обрабатываем каждый символ строки
    for i in range(1, n + 1):
        char = sequence[i - 1]

        for j in range(n):  # j - текущее количество открытых скобок
            if dp[i - 1][j] == 0:
                continue

            # Если символ — это "?", пробуем все скобки
            if char == '?':
                # Открывающие скобки
                if j + 1 <= n:
                    dp[i][j + 1] = (dp[i][j + 1] + dp[i - 1][j]) % MOD
                # Закрывающие скобки
                if j - 1 >= 0:
                    dp[i][j - 1] = (dp[i][j - 1] + dp[i - 1][j]) % MOD
            # Открывающие скобки
            elif char in "({[":
                if j + 1 <= n:
                    dp[i][j + 1] = (dp[i][j + 1] + dp[i - 1][j]) % MOD
            # Закрывающие скобки
            elif char == ')':
                if j - 1 >= 0:
                    dp[i][j - 1] = (dp[i][j - 1] + dp[i - 1][j]) % MOD
            elif char == ']':
                if j - 1 >= 0:
                    dp[i][j - 1] = (dp[i][j - 1] + dp[i - 1][j]) % MOD
            elif char == '}':
                if j - 1 >= 0:
                    dp[i][j - 1] = (dp[i][j - 1] + dp[i - 1][j]) % MOD

    # Ответ - это dp[n][0], когда обработаны все символы и все скобки закрыты
    return dp[n][0]


if __name__ == "__main__":
    with open("input.txt", "r") as f:
        n = int(f.readline().strip())
        sequence = f.readline().strip()

    result = count_valid_sequences(n, sequence)

    with open("output.txt", "w") as f:
        f.write(str(result) + "\n")
