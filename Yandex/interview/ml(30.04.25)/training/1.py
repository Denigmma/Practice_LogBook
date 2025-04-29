# 1. a * x**2 + b* x + c = 0
# найти градиентным спуском корни, данного уравнения
# (аналитически не решать, а то мы бы все просто дискриминант нашли).
# Обобщить решение на произвольную функцию, имеющую корни.
# Написать итеративный алгоритм на питоне для заданных a,b,c
# Как найти все корни?

import numpy as np

tol_grad=1e-8
x_bound=1e6
eps=1e-4
l_rate=0.0001
max_iter=10000
start_points=np.linspace(-20,20,500)

def loss(x,f):
    return f(x)**2

def dloss(x,f,df):
    return 2*f(x)*df(x)

def grad(xstart,f,df):
    x=xstart
    for _ in range(max_iter):
        grad=dloss(x,f,df)
        if abs(grad) < tol_grad :
            return None
        x=x-l_rate*grad
        if abs(x) > x_bound:
            return None
        if abs(f(x)) < 1e-8:
            return x
    return None


def sol(f,df):
    roots=[]
    for x in start_points:
        root=grad(x,f,df)
        if root is None:
            continue
        r=round(root,5)
        if not any(abs(r - existing) < eps for existing in roots):
            roots.append(r)
    print(f"Найдено корней: {len(roots)}")
    return roots


# a, b, c = 1, -3, 2
a, b, c,d = 3, -50, 2, 0

f = lambda x: x**3 - 1
df = lambda x: 3*x**2

# f = lambda x: a*x**2+b*x+c
# df = lambda x: 2*a*x+b

# f = lambda x: a*x**3+b*x**2+c*x+d
# df = lambda x: 3*a*x**2+2*b*x+c

print(sol(f,df))

