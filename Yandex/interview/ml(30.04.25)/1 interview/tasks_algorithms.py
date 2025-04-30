"""
Дан список целых чисел, повторяющихся элементов в списке нет.
Нужно преобразовать это множество в строку,
сворачивая соседние по числовому ряду числа в диапазоны.

Примеры:
- [1, 4, 5, 2, 3, 9, 8, 11, 0]  => "0-5,8-9,11"
- [1, 4, 3, 2]                   => "1-4"
- [1, 4]                         => "1,4"
"""


def compress(l):
    # your code here
    if not l:
        return ""
    sorted_nums = sorted(l)
    start = end = sorted_nums[0]
    parts = []
    for n in sorted_nums[1:]:
        if n == end + 1:
            end = n
        else:
            if start == end:
                parts.append(str(start))
            else:
                parts.append(f"{start}-{end}")
            start = end = n
    if start == end:
        parts.append(str(start))
    else:
        parts.append(f"{start}-{end}")
    return ",".join(parts)

print(compress([1, 4, 5, 2, 3, 9, 8, 11, 0]))  # "0-5,8-9,11"
print(compress([1, 4, 3, 2]))  # "1-4"
print(compress([1, 4]))  # "1,4"
