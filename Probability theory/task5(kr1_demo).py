import math


def combinations(n, k):
    """Вычисляет количество сочетаний C(n, k)"""
    if k > n or k < 0:
        return 0
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def probability(N, M=1000, n=150, k=10):
    """Вычисляет вероятность поймать k помеченных рыб среди n рыб."""
    p_marked = M / N  # вероятность поймать помеченную рыбу
    return combinations(n, k) * (p_marked ** k) * ((1 - p_marked) ** (n - k))


def find_max_probability():
    max_prob = 0
    best_N = 0

    for N in range(1000, 2000000):
        prob = probability(N)
        if prob > max_prob:
            max_prob = prob
            best_N = N

    return best_N, max_prob


best_N, max_prob = find_max_probability()
print(f"Лучшее количество рыб в озере: {best_N}")
print(f"Максимальная вероятность поймать 10 помеченных рыб: {max_prob:.6f}")
