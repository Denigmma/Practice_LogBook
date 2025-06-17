'''
Дан целочисленный массив nums, переместите все 0 в его конец, сохранив относительный порядок ненулевых элементов.

Обратите внимание, что вы должны сделать это на месте, не создавая копию массива.

Пример 1:

Ввод: nums = [0, 1, 0, 3, 12]
Вывод: [1, 3, 12, 0, 0]
Пример 2:

Ввод: nums = [0]
Вывод: [0]


Ограничения:

1 <= nums.length <= 104
-231 <= nums[i] <= 231 - 1


Последующие действия: Не могли бы вы свести к минимуму общее количество выполняемых операций?
'''


def moveZeroes(nums):
    """
    Do not return anything, modify nums in-place instead.
    """
    # for i in range(len(nums)-1):
    #     if nums[i]==0:
    #         point=i
    #         while point!=len(nums)-1:
    #             nums[point], nums[point + 1] = nums[point + 1], nums[point]
    #             point+=1

    nonzero=0
    for i in range(len(nums)):
        if nums[i]!=0:
            nums[nonzero], nums[i] = nums[i], nums[nonzero]
            nonzero += 1
    return nums


# Тренировка
def moveZeroes(nums):
    """
    Do not return anything, modify nums in-place instead.
    """
    nonezero=0
    for i in range(len(nums)):
        if nums[i]!=0:
            nums[nonezero],nums[i]=nums[i],nums[nonezero]
            nonezero+=1
    return nums


print(moveZeroes([5,0, 1, 0, 3, 12]))
print(moveZeroes([0]))










