'''
S. CUPED

Вам даны данные следующего формата: поюзерный исторический gmv (gmv_hist), поюзерный gmv во время эксперимента (gmv_exp), группа эксперимента (group_name).

Посчитайте коэффициент θ = Cov(X, Y) / Var X, используя все данные и теста, и контроля), где X - gmv_hist, Y - gmv_exp.
Найдите cuped значение метрики.
Посчитайте P-value без применения CUPED и P-value с применением CUPED
Используйте данные из файла synthetic_gmv_data_1.3.csv

Формат вывода
В ответе выведите два числа через пробел. Целую и дробную часть разделяйте точкой.

Примечания
Округлите до 3-го знака после точки.
'''

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

df = pd.read_csv('synthetic_gmv_data_1.3.csv')

# θ = Cov(X, Y) / Var(X)
X = df['gmv_hist']
Y = df['gmv_exp']
cov_xy = np.cov(X, Y, ddof=1)[0, 1]
var_x = X.var(ddof=1)
theta = cov_xy / var_x

# CUPED‑метрика
# Y_cuped = Y - θ * X
df['y_cuped'] = df['gmv_exp'] - theta * df['gmv_hist']

y_test = df.loc[df['group_name'] == 'test', 'gmv_exp']
y_control = df.loc[df['group_name'] == 'control', 'gmv_exp']

y_cuped_test = df.loc[df['group_name'] == 'test', 'y_cuped']
y_cuped_control = df.loc[df['group_name'] == 'control', 'y_cuped']

# p-value
# без CUPED
stat1, p1 = ttest_ind(y_test, y_control, equal_var=True)
# с CUPED
stat2, p2 = ttest_ind(y_cuped_test, y_cuped_control, equal_var=True)

print(f"{p1:.3f} {p2:.3f}")


# answer: 0.233 0.047