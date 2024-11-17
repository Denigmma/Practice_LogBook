# Чтение данных
with open('input.txt', 'r') as f:
    N = int(f.readline().strip())
    A = list(map(int, f.readline().strip().split()))
    B = list(map(int, f.readline().strip().split()))
    C = list(map(int, f.readline().strip().split()))

# Построение множеств событий для каждой цивилизации
set_A = set(A)
set_B = set(B)
set_C = set(C)

# Найти пересечение множеств
common_events = set_A & set_B & set_C

# Подсчитать количество элементов, которые нужно удалить
to_remove_A = len(set_A - common_events)
to_remove_B = len(set(B) - common_events)
to_remove_C = len(set(C) - common_events)

# Итоговое количество удалений
result = to_remove_A + to_remove_B + to_remove_C

# Запись результата в output.txt
with open('output.txt', 'w') as f:
    f.write(str(result) + '\n')
