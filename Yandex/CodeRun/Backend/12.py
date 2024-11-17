# ### С использованием scipy
# from scipy.integrate import quad
# n, m = map(int, input().split())
# lnn = list(map(int, input().split()))
# abcn=[]
# for _ in range(n):
#     a, b, c = map(int, input().split())
#     abcn.append((a, b, c))
# rmm = list(map(int, input().split()))
# abcm =[]
# for _ in range(m):
#     a, b, c = map(int, input().split())
#     abcm.append((a, b, c))
#
# def f(x,i):
#     return abcn[i][0] * x ** 2 + abcn[i][1] * x + abcn[i][2]
#
# def g(x,i):
#     return abcm[i][0]*x**2+abcm[i][1]*x+abcm[i][2]
#
# def integrate():
#     total_area = 0.0
#     i, j = 0, 0
#     while i < n and j < m:
#         l_start, l_end = lnn[i], lnn[i + 1]
#         r_start, r_end = rmm[j], rmm[j + 1]
#         start = max(l_start, r_start)
#         end = min(l_end, r_end)
#
#         if start < end:
#             def fg(x):
#                 return abs(f(x,i) - g(x,j))
#             area, _ = quad(fg, start, end)
#             total_area += area
#         if l_end < r_end:
#             i += 1
#         else:
#             j += 1
#
#     return total_area
#
# print(f"{integrate():.6f}")



n, m = map(int, input().split())
lnn = list(map(int, input().split()))
abcn=[]
for _ in range(n):
    a, b, c = map(int, input().split())
    abcn.append((a, b, c))
rmm = list(map(int, input().split()))
abcm =[]
for _ in range(m):
    a, b, c = map(int, input().split())
    abcm.append((a, b, c))

def f(x,i):
    return abcn[i][0] * x ** 2 + abcn[i][1] * x + abcn[i][2]

def g(x,i):
    return abcm[i][0]*x**2+abcm[i][1]*x+abcm[i][2]

# Метод трапеций
def trapezoidal_integration(func, a, b, num_steps=1000):
    h = (b - a) / num_steps  # шаг
    integral = 0.5 * (func(a) + func(b))
    x = a
    for _ in range(1, num_steps):
        x += h
        integral += func(x)
    return integral * h

def integrate():
    total_area = 0.0
    i, j = 0, 0
    while i < n and j < m:
        l_start, l_end = lnn[i], lnn[i + 1]
        r_start, r_end = rmm[j], rmm[j + 1]
        start = max(l_start, r_start)
        end = min(l_end, r_end)

        if start < end:
            def fg(x):
                return abs(f(x,i) - g(x,j))
            area = trapezoidal_integration(fg, start, end)
            total_area += area
        if l_end < r_end:
            i += 1
        else:
            j += 1

    return total_area

print(f"{integrate():.6f}")