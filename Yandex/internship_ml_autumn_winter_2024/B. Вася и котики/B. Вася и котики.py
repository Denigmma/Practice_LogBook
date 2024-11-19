def can_place_cats(lying_beds, occupied_beds, n, d):
    """Проверяем, можем ли мы разместить всех котиков с минимальным расстоянием d."""
    # Доступные лежанки - это все лежанки минус уже занятые
    available_beds = sorted(set(lying_beds) - set(occupied_beds))

    cats_count = len(occupied_beds)  # Начинаем с котов, которые уже на занятых лежанках
    last_position = occupied_beds[-1]  # Начинаем с последней занятой лежанки

    # Перебираем все доступные лежанки
    for bed in available_beds:
        if bed >= last_position + d:  # Если расстояние достаточно
            cats_count += 1
            last_position = bed  # Последняя занятная лежанка - это текущая
            if cats_count == n:  # Если всех котиков разместили
                return True
    return False


def solve(n, m, k, lying_beds, occupied_beds):
    lying_beds.sort()  # Сортируем все лежанки
    occupied_beds.sort()  # Сортируем уже занятые лежанки

    # Бинарный поиск по минимальному счастью (расстоянию)
    left = 0  # минимальное расстояние
    right = lying_beds[-1] - lying_beds[0]  # максимальное расстояние (между самой левой и правой лежанками)

    best_d = 0

    while left <= right:
        mid = (left + right) // 2
        if can_place_cats(lying_beds, occupied_beds, n, mid):
            best_d = mid  # нашли подходящее минимальное счастье
            left = mid + 1  # пробуем большее расстояние
        else:
            right = mid - 1  # пробуем меньшее расстояние

    return best_d


# Ввод
n, m, k = 2, 4, 1
lying_beds = [1, 5, 6, 9]
occupied_beds = [6]

# Решение
result = solve(n, m, k, lying_beds, occupied_beds)
print(result)
