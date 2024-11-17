import math
import time


# Функция для проверки числа на простоту
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


# Функция для подсчета количества делителей числа
def count_divisors(n):
    count = 0
    sqrt_n = int(math.sqrt(n))
    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count


# Основная функция для подсчета чисел с простым количеством делителей
def count_composite_numbers_with_prime_divisors(l, r):
    count = 0
    limit = int(math.sqrt(r))  # Ограничение до корня из r

    # Перебираем все простые числа до sqrt(r)
    for p in range(2, limit + 1):
        if is_prime(p):
            p2 = p * p  # Начинаем с квадрата простого числа
            if p2 > r:
                break

            # Увеличиваем кратные p^2 и проверяем их в диапазоне
            x = p2
            while x <= r:
                if x >= l and is_prime(count_divisors(x)):
                    count += 1
                # Переходим к следующей степени p^2
                if x > r // p2:
                    break
                x *= p2

    return count


# Чтение входных данных
l, r = map(int, input().split())

# Запуск таймера
start_time = time.time()

# Вывод результата
print(count_composite_numbers_with_prime_divisors(l, r))

# Остановка таймера и вывод времени выполнения
end_time = time.time()
execution_time = end_time - start_time
print(f"Время выполнения: {execution_time:.2f} секунд")
