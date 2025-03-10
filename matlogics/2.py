def implication(A, B):
    """Функция для логической импликации (A ⊃ B), эквивалентной (not A or B)"""
    return (not A) or B

def check_tautology(formula):
    """Проверяет, является ли формула тождественно истинной."""
    for Q0 in [False, True]:
        for Q1 in [False, True]:
            for Q2 in [False, True]:
                if not formula(Q0, Q1, Q2):
                    return False  # Нашли контрпример, не тождественная истина
    return True  # Во всех случаях истина

# Формула (a)
def formula_a(Q0, Q1, Q2):
    return implication(
        implication(Q0, Q1),
        implication(
            implication(Q0, implication(Q1, Q2)),
            implication(Q0, Q2)
        )
    )

# Формула (b)
def formula_b(Q0, Q1, Q2):
    return implication(
        implication(Q0, Q1),
        implication(
            (Q0 or Q2),
            implication(Q1, Q2)
        )
    )

# Проверяем тождественную истинность формул
print("Формула (a) тождественно истинна:", check_tautology(formula_a))
print("Формула (b) тождественно истинна:", check_tautology(formula_b))
