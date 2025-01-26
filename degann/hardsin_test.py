import numpy as npt
import matplotlib.pyplot as pl

# Определяем функцию
def hard_sin(x):
    return np.sin(np.log(x**np.sin(15*x)))

# Создаем массив x с шагом для избежания нуля
x = np.linspace(0.01, 2, 1000)  # начиная с 0.01, чтобы избежать log(0)
y = hard_sin(x)

# Рисуем график
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\sin(\ln(x^{\sin(x)}))$', color='blue')
plt.title('График функции $\\sin(\\ln(x^{\\sin(x)}))$', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()
