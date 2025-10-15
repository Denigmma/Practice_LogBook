import re

best_util = float('-inf')
best_lambda = None


with open("сетка.txt", "r", encoding="utf-8") as f:
    for line in f:
        match = re.search(r"λ=\(([^)]+)\): util=([0-9.]+)", line)
        if match:
            lambdas_str = match.group(1)
            util_val = float(match.group(2))
            lambdas = tuple(float(x.strip()) for x in lambdas_str.split(','))
            if util_val > best_util:
                best_util = util_val
                best_lambda = lambdas

print("📈 Лучшая λ:", best_lambda)
print("⭐️ Лучший util:", best_util)
