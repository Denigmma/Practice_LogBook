'''
Учитывая массив целых чисел nums и целое число k, верните общее количество подмассивов, сумма которых равна k.

Подмассив — это непрерывная непустая последовательность элементов внутри массива.

Пример 1:
Ввод: nums = [1,1,1], k = 2
Вывод: 2

Пример 2:
Ввод: nums = [1,2,3], k = 3
Вывод: 2

Ограничения:

1 <= nums.length <= 2 * 104
-1000 <= nums[i] <= 1000
-107 <= k <= 107
'''
from typing import List
from collections import defaultdict


class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        count = 0
        current_sum = 0
        prefix_sums = defaultdict(int)
        prefix_sums[0] = 1

        for num in nums:
            current_sum += num
            if (current_sum - k) in prefix_sums:
                count += prefix_sums[current_sum - k]
            prefix_sums[current_sum] += 1
        return count



class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        prefix_sums = {0: 1}  # сумма 0 встречается один раз перед началом массива
        current_sum = 0
        result = 0

        for num in nums:
            current_sum += num
            if current_sum - k in prefix_sums:
                result += prefix_sums[current_sum - k]
            if current_sum in prefix_sums:
                prefix_sums[current_sum] += 1
            else:
                prefix_sums[current_sum] = 1

        return result


solution = Solution()
example =[1,1,1]
k=2
print(solution.subarraySum(example,k))
