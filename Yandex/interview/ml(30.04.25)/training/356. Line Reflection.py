'''
Даны n точки на двумерной плоскости. Найдите, существует ли такая линия, параллельная оси y,
которая симметрично отражает заданные точки.

Другими словами, ответьте, существует ли такая линия, что после отражения всех
 точек относительно этой линии исходный набор точек будет таким же, как и отражённые точки.

Обратите внимание, что точки могут повторяться.

Пример 1:

Входные данные: точки = [[1,1],[-1,1]]
Выходные данные: true
Объяснение: Мы можем выбрать прямую x = 0.

Пример 2:

Входные данные: точки = [[1,1],[-1,-1]]
Выходные данные: false
Объяснение: Мы не можем выбрать линию.
'''
from typing import List

### Решение 1 за O(n log n)
# class Solution:
#     def isReflected(self, points: List[List[int]]) -> bool:
#         dict_y_to_x = {}  # все y и перечисление [х]
#         mid = None
#
#         # находим x точек, которые лежат на одном y
#         for x, y in points:
#             if y not in dict_y_to_x:
#                 dict_y_to_x[y] = []
#             dict_y_to_x[y].append(x)
#
#         print(dict_y_to_x) # {1: [-5, 5], 4: [-3, 3], 0: [1]}
#
#         for y in dict_y_to_x:
#             x_list = dict_y_to_x[y]
#             x_list.sort()  # O(k log k), где k — количество точек на этой прямой y
#
#             left, right = 0, len(x_list) - 1
#             while left < right:
#                 current_mid = (x_list[left] + x_list[right]) / 2
#                 if mid is None:
#                     mid = current_mid  # устанавливаем первую середину
#                 elif current_mid != mid:
#                     # print(f"Несовпадение середин: {current_mid} != {mid}")
#                     return False
#                 left += 1
#                 right -= 1
#
#             # если осталась одиночная точка на этой прямой
#             if left == right:
#                 single_x = x_list[left]
#                 if mid is None:
#                     mid = single_x  # если это первая точка вообще
#                 elif single_x != mid:
#                     return False
#         # print("Все середины совпали.")
#         return True


### Решение 2 за O(n)
class Solution:
    def isReflected(self, points: List[List[int]]) -> bool:
            min_x, max_x = float("inf"), float("-inf")
            point_set = set()
            for x, y in points:
                point_set.add((x, y))
                min_x=min(min_x,x)
                max_x = max(max_x, x)
            mid = (min_x + max_x) / 2
            for x, y in points:
                reflected_x = 2 * mid - x
                if (reflected_x, y) not in point_set:
                    return False
            return True


solution=Solution()
list=[[5,1],[-3,4],[3,4],[-5,1],[1,0]]
# list=[[1,1],[-1,1]]
print(solution.isReflected(list))


# class Solution:
#     def isReflected(self, points: List[List[int]]) -> bool:
#
#
# solution=Solution()
# list=[[5,1],[-3,4],[3,4],[-5,1],[1,0]]
# # list=[[1,1],[-1,1]]
# print(solution.isReflected(list))