def min_cost_for_magnets(n, k, magnets):
    from collections import defaultdict

    # Словарь для хранения количества магнитов
    count = defaultdict(int)

    l = 0  # левый указатель
    min_cost = float('inf')  # начальная минимальная стоимость
    total_cost = 0  # текущая стоимость подотрезка
    distinct_count = 0  # количество уникальных магнитов в подотрезке

    for r in range(n):  # правый указатель
        # Добавляем магнит из правой части
        magnet_type = magnets[r]
        total_cost += magnet_type

        if 1 <= magnet_type <= k:
            count[magnet_type] += 1
            if count[magnet_type] == 1:  # впервые встретили данный магнит
                distinct_count += 1

        # Уменьшаем окно с левой стороны, если все типы есть
        while distinct_count == k:
            min_cost = min(min_cost, total_cost)
            left_magnet_type = magnets[l]
            total_cost -= left_magnet_type

            if 1 <= left_magnet_type <= k:
                count[left_magnet_type] -= 1
                if count[left_magnet_type] == 0:  # магнит больше не в окне
                    distinct_count -= 1

            l += 1  # сжимаем окно

    return min_cost


# Чтение данных
n, k = map(int, input().strip().split())
magnets = list(map(int, input().strip().split()))

# Получение результата и вывод
result = min_cost_for_magnets(n, k, magnets)
print(result)
