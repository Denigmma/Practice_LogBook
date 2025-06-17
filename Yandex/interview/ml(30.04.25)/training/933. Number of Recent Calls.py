'''
У вас есть класс RecentCounter , который подсчитывает количество недавних запросов за определённый период времени.

Реализовать класс RecentCounter:

RecentCounter() Инициализирует счетчик нулевыми последними запросами.
int ping(int t) Добавляет новый запрос в момент времени t, где t обозначает время в миллисекундах, и возвращает количество запросов, выполненных за последние 3000 миллисекунд (включая новый запрос). В частности, возвращает количество запросов, выполненных в диапазоне [t - 3000, t].
Гарантируется, что при каждом вызове ping используется строго большее значение t, чем при предыдущем вызове.



Пример 1:

Входные данные
["RecentCounter", "ping", "ping", "ping", "ping"]
[[], [1], [100], [3001], [3002]]
Вывод
[null, 1, 2, 3, 3]

Объяснение
RecentCounter recentCounter = new RecentCounter();
recentCounter.ping(1); // запросы = [1], диапазон равен [-2999,1], возвращает 1
recentCounter.ping(100); // запросы = [1, 100], диапазон равен [-2900,100], возврат 2
recentCounter.ping(3001); // запросы = [1, 100, 3001], диапазон равен [1,3001], возврат 3
recentCounter.ping(3002); // запросы = [1, 100, 3001, 3002], диапазон равен [23002], возврат 3


Ограничения:

1 <= t <= 109
Каждый тестовый пример будет вызывать ping со строго возрастающими значениями t.
Не более 104 звонков будет сделано по адресу ping.
'''

from collections import deque

class RecentCounter:

    def __init__(self):
        self.requests = deque()

    def ping(self, t: int) -> int:
        self.requests.append(t)
        while self.requests[0] < t - 3000:
            self.requests.popleft()
        return len(self.requests)

# Your RecentCounter object will be instantiated and called as such:
# obj = RecentCounter()
# param_1 = obj.ping(t)