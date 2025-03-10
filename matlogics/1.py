import itertools

# Определяем функцию для вычисления A
def compute_A(Q0, Q1, Q2, Q3):
    return (Q2 and not Q3) or (not Q0 and Q1 and Q2 and Q3) or (Q0 and Q1 and not Q2 )

# Данные из таблицы истинности
truth_table = [
    (0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0),
    (0, 0, 1, 0, 1),
    (0, 0, 1, 1, 0),
    (0, 1, 0, 0, 0),
    (0, 1, 0, 1, 0),
    (0, 1, 1, 0, 1),
    (0, 1, 1, 1, 1),
    (1, 0, 0, 0, 0),
    (1, 0, 0, 1, 0),
    (1, 0, 1, 0, 1),
    (1, 0, 1, 1, 0),
    (1, 1, 0, 0, 1),
    (1, 1, 0, 1, 1),
    (1, 1, 1, 0, 1),
    (1, 1, 1, 1, 0),
]

# Проверим, совпадает ли вычисленное значение A с таблицей
errors = []
for Q0, Q1, Q2, Q3, expected_A in truth_table:
    computed_A = int(compute_A(Q0, Q1, Q2, Q3))
    if computed_A != expected_A:
        errors.append((Q0, Q1, Q2, Q3, expected_A, computed_A))

print(errors)
