'''
Q. Линеаризация второго типа

На тех же данных теперь проверим гипотезу равенства среднего чека с использованием линеаризации второго типа и классического t‑теста.

Перейдём к линеаризованной метрике
L(u) = X̄/Ȳ + (1/Ȳ)·X(u) − (X̄/Ȳ²)·Y(u),

где
X(u) — значение числителя для пользователя u
Y(u) — значение знаменателя для пользователя u
X̄ — среднее в группе по числителю
Ȳ — среднее в группе по знаменателю

Посчитайте T‑статистику, найдите P‑value для линеаризованной метрики.
Используйте данные из файла synthetic_gmv_data_1.2.csv

Формат вывода
В ответе выведите два числа через пробел. Целую и дробную часть разделяйте точкой.
Пример ответа: -0.046 0.476

Примечания
Округлите до 3-го знака после точки.
'''


import pandas as pd
import os
from scipy.stats import ttest_ind

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, '..', 'O. t-test', 'synthetic_gmv_data_1.2.csv')

df = pd.read_csv(data_path)
# агрегируем: X = сумма gmv, Y = число транзакций на пользователя
df_user = df.groupby(['user_id', 'group_name'], as_index=False) \
            .agg(X=('gmv', 'sum'), Y=('gmv', 'size'))

# разбиваем на контроль и тест
ctrl = df_user[df_user['group_name']=='control'].copy()
tst  = df_user[df_user['group_name']=='test'].copy()

# считаем групповые средние
Xc_bar = ctrl['X'].mean()
Yc_bar = ctrl['Y'].mean()
Xt_bar = tst ['X'].mean()
Yt_bar = tst ['Y'].mean()

# линеаризация по формуле
ctrl['L'] = Xc_bar/Yc_bar + ctrl['X']/Yc_bar - Xc_bar/(Yc_bar**2)*ctrl['Y']
tst ['L'] = Xt_bar/Yt_bar + tst ['X']/Yt_bar - Xt_bar/(Yt_bar**2)*tst ['Y']

# T‑тест для L
t_stat, p_value = ttest_ind(tst['L'], ctrl['L'], equal_var=False)

print(f"{t_stat:.3f} {p_value:.3f}")


# answer: 2.344 0.019