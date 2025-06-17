"""
1 строка из символов
2 найти кол-во пар индексов i и j между которыми нет повторяющихся символов
3 (подстрока, где все уникальны)

abca -> 9
bc
b
c
abc
bca
ab
ca
a
a

# Для строки "aba" ответ 5:
# [0, 0] ("a")
# [0, 1] ("ab")
# [1, 1] ("b")
# [1, 2] ("ba")
# [2, 2] ("a")
"""

"""
похожа на LeetCode: 3 — Longest Substring Without Repeating Characters
"""

def count_unique_substrings(s):
    m = set()
    left = 0
    result = 0

    for right in range(len(s)):
        while s[right] in m:
            m.remove(s[left])
            left += 1
        m.add(s[right])
        result += right - left + 1
    return result
