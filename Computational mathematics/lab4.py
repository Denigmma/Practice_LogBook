import sympy as sp

# Определение переменной x
x = sp.symbols('x')

# Функция для вычисления квадратурной формулы Гаусса с двумя узлами
def quadrature_formula_Gauss(p_x, a, b):
    # Моменты интегралов
    c0 = sp.integrate(p_x, (x, a, b))
    c1 = sp.integrate(p_x * x, (x, a, b))
    c2 = sp.integrate(p_x * x**2, (x, a, b))
    c3 = sp.integrate(p_x * x**3, (x, a, b))

    # Символьные переменные для коэффициентов
    s, r = sp.symbols('s r')

    # Составляем систему уравнений для нахождения коэффициентов s и r
    eq1 = sp.Eq(c2 + s * c1 + r * c0, 0)
    eq2 = sp.Eq(c3 + s * c2 + r * c1, 0)

    # Решение системы уравнений для s и r
    solution = sp.solve([eq1, eq2], (s, r))
    s_val = solution[s]
    r_val = solution[r]

    # Корни многочлена x1 и x2
    omega = x**2 + s_val * x + r_val
    roots = sp.solve(omega, x)
    x1, x2 = roots

    # Символьные переменные для A1 и A2
    A1, A2 = sp.symbols('A1 A2')

    # Уравнения для нахождения A1 и A2
    A1_A2_eq1 = sp.Eq(A1 + A2, c0)
    A1_A2_eq2 = sp.Eq(A1 * x1 + A2 * x2, c1)

    # Решение системы уравнений для A1 и A2
    A_solution = sp.solve([A1_A2_eq1, A1_A2_eq2], (A1, A2))
    A1_val = A_solution[A1]
    A2_val = A_solution[A2]

    # Возвращаем корни и коэффициенты A1 и A2
    return A1_val, A2_val, x1, x2

# Форматирование результата в виде квадратурной формулы
def format_quadrature(A1, A2, x1, x2):
    return f"{A1} * f({x1}) + {A2} * f({x2})"

# Решение для задачи A (различные значения alpha)
A1_A2_x1_x2_1_2 = quadrature_formula_Gauss(x ** (1 / 2), 0, 1)
A1_A2_x1_x2_2 = quadrature_formula_Gauss(x ** (2), 0, 1)
A1_A2_x1_x2_3 = quadrature_formula_Gauss(x ** (3), 0, 1)

# Решение для задачи Б (p(x) = e^x)
A1_A2_x1_x2_B = quadrature_formula_Gauss(sp.exp(x), 0, 0.5)

# Решение для задачи В (p(x) = ln(x))
A1_A2_x1_x2_C = quadrature_formula_Gauss(sp.ln(x), 0.5, 1)

# Вывод результатов в требуемом формате
print("Result for alpha = 1/2:", format_quadrature(*A1_A2_x1_x2_1_2))
print("Result for alpha = 2:", format_quadrature(*A1_A2_x1_x2_2))
print("Result for alpha = 3:", format_quadrature(*A1_A2_x1_x2_3))
print("Result for p(x) = e^x:", format_quadrature(*A1_A2_x1_x2_B))
print("Result for p(x) = ln(x):", format_quadrature(*A1_A2_x1_x2_C))
