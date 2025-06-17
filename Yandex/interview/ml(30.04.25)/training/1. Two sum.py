'''
Учитывая массив целых чисел nums и целое число target, верните индексы двух чисел, сумма которых равна target.
Вы можете предположить, что для каждого ввода будет ровноодно решение,
и вы не можете использовать один и тот же элемент дважды.

Пример 1:

Ввод: nums = [2,7,11,15], target = 9
Вывод: [0,1]
Объяснение: Поскольку nums[0] + nums[1] == 9, мы возвращаем [0, 1].
Пример 2:

Ввод: nums = [3,2,4], target = 6
Вывод: [1,2]
Пример 3:

Ввод: nums = [3,3], target = 6
Вывод: [0,1]

Ограничения:
2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Существует только один правильный ответ.


Продолжение: Можете ли вы придумать алгоритм, который будет работать быстрее, чем O(n2) временная сложность?
'''

def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    dict = {}
    for i, num in enumerate(nums):
        a = target - num
        if a in dict:
            return [dict[a], i]
        dict[num] = i
    return []

nums=[3,2,4]
target = 6
print(twoSum(nums,target))




# Тренировка
def twoSum(nums, target):
    dict={}
    for i in range(len(nums)):
        diff=target-nums[i]
        if nums[i] in dict:
            return [dict[nums[i]],i]
        dict[diff]=i


nums=[2,7,11,15]
target = 9
print(twoSum(nums,target))