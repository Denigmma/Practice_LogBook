'''
Мощность строки — это максимальная длина непустой подстроки, содержащей только один уникальный символ.

Учитывая строку s, верните мощность из s.



Пример 1:

Ввод: s = «leetcode»
Вывод: 2
Объяснение: Подстрока «ee» имеет длину 2 и состоит только из символа «e».
Пример 2:

Ввод: s = «abbcccddddeeeeedcba»
Вывод: 5
Пояснение: Подстрока «eeeee» имеет длину 5 и состоит только из символа «e».


Ограничения:

1 <= s.length <= 500
s состоит только из строчных английских букв.
'''


def maxPower(s):
    """
    :type s: str
    :rtype: int
    """
    max_len = 1
    lenth = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            lenth += 1
        else:
            max_len = max(lenth, max_len)
            lenth = 1
    return max(lenth, max_len)



def maxPower(s):
    """
    :type s: str
    :rtype: int
    """
    max_power = 1
    current_power = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            current_power += 1
            max_power = max(max_power, current_power)
        else:
            current_power = 1
    return max_power


print(maxPower("leetcode"))