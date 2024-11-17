import math
import time

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def count_divisors(n):
    count = 0
    sqrt_n = int(math.sqrt(n))
    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count

def count_composite_numbers_with_prime_divisors(l, r):
    primes = [p for p in range(2, int(math.sqrt(r)) + 1) if is_prime(p)]
    count = 0
    for p in primes:
        p2 = p * p
        if p2 > r:
            break
        for k in range(1, 11):  # We limit exponent to ensure we are within bounds
            p_k = p2 ** k
            if p_k > r:
                break
            if p_k >= l and is_prime(count_divisors(p_k)):
                count += 1
    return count


# Read input
l, r = map(int, input().split())

# Запуск таймера
start_time = time.time()

print(count_composite_numbers_with_prime_divisors(l, r))

# Остановка таймера и вывод времени выполнения
end_time = time.time()
execution_time = end_time - start_time
print(f"Время выполнения: {execution_time:.2f} секунд")
