'''
Вам предоставляется отсортированный уникальный целочисленный массив nums.

Диапазон [a,b] — это множество всех целых чисел от a до b (включительно).

Возвращает наименьший отсортированный список диапазонов, которые точно охватывают все числа в массиве. То есть каждый элемент nums покрывается ровно одним из диапазонов, и не существует целого числа, x такого, x которое находится в одном из диапазонов, но не в nums.

Каждый диапазон [a,b] в списке должен быть выведен в виде:

"a->b" если a != b
"a" если a == b


Пример 1:

Ввод: nums = [0,1,2,4,5,7]
Вывод: ["0->2","4->5","7"]
Пояснение: Диапазоны:
[0,2] --> "0->2"
[4,5] --> "4->5"
[7,7] —> «7»
Пример 2:

Ввод: nums = [0, 2, 3, 4, 6, 8, 9]
Вывод: ["0", "2->4", "6", "8->9"]
Объяснение: Диапазоны:
[0,0] --> "0"
[2,4] --> "2->4"
[6,6] --> "6"
[8,9] --> "8->9"


Ограничения:

0 <= nums.length <= 20
-231 <= nums[i] <= 231 - 1
Все значения nums являются уникальными.
nums сортируется в порядке возрастания.
'''

### O(n) - время, O(n) - память
def summaryRanges(nums):
    """
    :type nums: List[int]
    :rtype: List[str]
    """
    array = []
    count=0

    for i in range(len(nums)):
        if i==len(nums)-1:
            if count==0:
                array.append(f"{nums[i]}")
            else:
                array.append(f"{nums[i - count]}->{nums[i]}")
        else:
            if nums[i]+1 == nums[i+1]:
                count+=1
            else:
                if count==0:
                    array.append(f"{nums[i]}")
                else:
                    array.append(f"{nums[i-count]}->{nums[i]}")
                count=0
    return array


print(summaryRanges([0,1,2,4,5,7]))
print(summaryRanges([0, 2, 3, 4, 6, 8, 9]))


# Тренировка
def summaryRanges(nums):
    """
    :type nums: List[int]
    :rtype: List[str]
    """
    array=[]
    last=0
    for i in range(len(nums)):
        if i==len(nums)-1:
            if last==i:
                array.append(f"{nums[i]}")
            else:
                array.append(f"{nums[last]}->{nums[i]}")
        else:
            if nums[i]+1!=nums[i+1]:
                if last!=i:
                    array.append(f"{nums[last]}->{nums[i]}")
                    last=i+1
                else:
                    array.append(f"{nums[i]}")
                    last=i+1
    return array

print(summaryRanges([0,1,2,4,5,7]))
print(summaryRanges([0, 2, 3, 4, 6, 8, 9]))