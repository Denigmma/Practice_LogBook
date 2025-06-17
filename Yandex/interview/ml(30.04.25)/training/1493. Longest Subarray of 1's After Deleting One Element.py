'''
Учитывая двоичный массив nums, вы должны удалить из него один элемент.

Верните размер самого длинного непустого подмассива, содержащего только 1's в результирующем массиве. Верните 0, если такого подмассива нет.



Пример 1:

Ввод: nums = [1,1,0,1]
Вывод: 3
Пояснение: После удаления числа в позиции 2 [1,1,1] содержит 3 числа со значением 1.
Пример 2:

Ввод: nums = [0,1,1,1,0,1,1,0,1]
Вывод: 5
Объяснение: После удаления числа в позиции 4 [0,1,1,1,1,1,0,1] самый длинный подмассив со значением 1 — [1,1,1,1,1].
Пример 3:

Ввод: nums = [1,1,1]
Вывод: 2
Объяснение: Вы должны удалить один элемент.


Ограничения:

1 <= nums.length <= 105
nums[i] является либо 0, либо 1.
'''
from typing import List

### O(n) по вермени и O(1) по памяти
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        left=0
        max_count=0
        zero_count = 0

        for right in range(len(nums)):
            if nums[right] == 0:
                zero_count += 1

            while zero_count > 1:
                if nums[left] == 0:
                    zero_count -= 1
                left += 1

            max_count = max(max_count, right - left)

        # for right in range(len(nums)):
        #     if 0 in nums[left: right + 1]:
        #         zero_count=nums[left: right + 1].count(0)
        #         if zero_count>1:
        #             left+=1
        #         else:
        #             max_count=max(len(nums[left: right + 1])-1,max_count)
        #     else:
        #         max_count=max(len(nums[left: right + 1])-1,max_count)

        return max_count




# Тренировка
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        max_lenth=0
        left=0
        zero_count=0

        for right in range(len(nums)):
            if nums[right]==0:
                zero_count+=1
                while zero_count>1:
                    if nums[left]==0:
                        zero_count-=1
                    left+=1
            max_lenth=max(max_lenth,right-left)
        return max_lenth


solution=Solution()
print(solution.longestSubarray([1,1,0,1]))
print(solution.longestSubarray([0,1,1,1,0,1,1,0,1]))
print(solution.longestSubarray([1,1,1]))
