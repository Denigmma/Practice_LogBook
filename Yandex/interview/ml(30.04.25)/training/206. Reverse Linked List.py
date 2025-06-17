'''
Учитывая head односвязного списка, переверните его и верните перевёрнутый список.

Пример 1:

Ввод: head = [1,2,3,4,5]
Вывод: [5,4,3,2,1]
Пример 2:

Ввод: head = [1,2]
Вывод: [2,1]
Пример 3:

Ввод: head = []
Результат: []

Ограничения:
Количество узлов в списке - это диапазон [0, 5000].
-5000 <= Node.val <= 5000

Продолжение: Связанный список можно перевернуть либо итеративно, либо рекурсивно. Не могли бы вы реализовать оба варианта?
'''
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head

        while curr:
            next_node = curr.next  # сохраняем ссылку на следующий
            curr.next = prev  # меняем направление ссылки
            prev = curr  # двигаем prev вперёд
            curr = next_node  # двигаем curr вперёд

        return prev





# Функция для вывода списка на экран
def print_list(head):
    current = head
    while current:
        print(current.val, end=" -> ")
        current = current.next
    print("None")

# Создаём связанный список 1 -> 2 -> 3 -> None
node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(3)

node1.next = node2
node2.next = node3

# Выводим оригинальный список
print("Исходный список:")
print_list(node1)

# Переворачиваем список
solution = Solution()
reversed_head = solution.reverseList(node1)

# Выводим перевёрнутый список
print("Перевёрнутый список:")
print_list(reversed_head)

# def reverseList(head):
#     """
#     :type head: Optional[ListNode]
#     :rtype: Optional[ListNode]
#     """
#     for i in range(len(head)//2):
#         head[i],head[len(head)-1-i]=head[len(head)-1-i],head[i]
#     return head
#
# array=[1,2,3,4,5]
# print(reverseList(array))