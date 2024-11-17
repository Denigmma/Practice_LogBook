import math


# Функция для проверки, является ли число простым
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


# Функция для подсчета делителей числа
def count_divisors(n):
    count = 0
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count


# Основная функция для подсчета нужных чисел
def count_special_squares(l, r):
    count = 0
    # Перебираем все целые числа, квадраты которых лежат в пределах от l до r
    start = int(math.ceil(math.sqrt(l)))  # Начало диапазона
    end = int(math.floor(math.sqrt(r)))  # Конец диапазона

    for i in range(start, end + 1):
        square = i * i
        divisors_count = count_divisors(square)

        # Проверяем, является ли количество делителей простым числом
        if is_prime(divisors_count):
            count += 1

    return count


# Ввод данных
l, r = map(int, input().split())

# Вывод результата
print(count_special_squares(l, r))
