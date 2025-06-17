'''
Дано: строка s, содержащая только символы '(', ')', '{', '}', '[' и ']'.
Определите, является ли входная строка допустимой.

Входная строка допустима, если:
Открытые скобки должны быть закрыты скобками того же типа.
Открытые скобки должны быть закрыты в правильном порядке.
Каждой закрывающей скобке соответствует открывающая скобка того же типа.

Пример 1:
Введите: s = «()»
Вывод: верно

Пример 2:
Ввод: s = «()[]{}»
Вывод: верно

Пример 3:
Введите: s = "(]"
Вывод: false

Пример 4:
Входные данные: s = "([])"
Вывод: верно


Ограничения:
1 <= s.length <= 104
s состоит только из круглых скобок '()[]{}'.
'''

class Solution:
    def isValid(self, s: str) -> bool:
        stack=[]
        dict={")":"(", "}":"{", "]":"["}
        for char in s:
            if char in "({[":
                stack.append(char)
            else:
                if not stack or stack[-1] != dict[char]:
                    return False
                stack.pop()
        return not stack



solution=Solution()
print(solution.isValid("()"))
print(solution.isValid("()[]{}"))
print(solution.isValid("(]"))
print(solution.isValid("([])"))
print(solution.isValid("([)]"))
print(solution.isValid("["))
print(solution.isValid("(("))
print(solution.isValid("]"))