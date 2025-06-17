'''
Разработайте структуру данных, которая соответствует ограничениям кэша с наименьшим временем использования (LRU).

Реализовать класс LRUCache:

LRUCache(int capacity) Инициализируйте кэш LRU с положительным размером capacity.

int get(int key) Верните значение key, если ключ существует, в противном случае верните -1.

void put(int key, int value) Обновите значение key если key существует.
В противном случае добавьте пару key-value в кэш. Если количество ключей превышает capacity
в результате этой операции, удалите ключ, который использовался последним.
Каждая из функций get и put должна выполняться со O(1) средней временной сложностью.


Example 1:

Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Объяснение
LRUCache lRUCache = новый LRUCache(2);
lRUCache.put(1, 1); // кэш {1=1}
lRUCache.put(2, 2); // кэш {1=1, 2=2}
lRUCache.get(1); // возвращает 1
lRUCache.put(3, 3); // ключ LRU был 2, вытесняет ключ 2, кэш {1=1, 3=3}
lRUCache.get(2); // возвращает -1 (не найдено)
lRUCache.put(4, 4); // ключ LRU был 1, вытесняет ключ 1, кэш {4=4, 3=3}
lRUCache.get(1); // возвращает -1 (не найдено)
lRUCache.get(3); // возвращает 3
lRUCache.get(4); // возвращает 4


Ограничения:

1 <= capacity <= 3000
0 <= key <= 104
0 <= value <= 105
Не более 2 * 105 звонков будет сделано на get и put.
'''

class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity=capacity
        self.dict={}

        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key in self.dict:
             # переместить ключ в конец списка (как «только что использованный»)
             node = self.dict[key]
             self._remove(node)
             self._add(node)
             return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.dict:
            node = self.dict[key]
            node.value = value
            # перемести в конец списка
            self._remove(node)
            self._add(node)
        else:
            '''
            если кэш полон:
                удали самый старый (из начала списка)
                удали его из словаря
            добавь новый элемент в конец списка и словарь
            '''
            if len(self.dict)>=self.capacity:
                lru_node = self.head.next
                self._remove(lru_node)
                del self.dict[lru_node.key]

            new_node = Node(key, value)
            self._add(new_node)
            self.dict[key] = new_node

    # Добавить узел в конец (перед tail)
    def _add(self, node):
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    # Удалить узел
    def _remove(self, node):
        prev = node.prev
        nxt = node.next
        prev.next = nxt
        nxt.prev = prev


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

# создаём объект LRUCache с capacity 2
lruCache = LRUCache(2)

# последовательно выполняем операции, как в примере
print(lruCache.put(1, 1))  # кэш: {1=1}
print(lruCache.put(2, 2))  # кэш: {1=1, 2=2}
print(lruCache.get(1))  # возвращает 1
print(lruCache.put(3, 3))  # удаляет ключ 2, кэш: {1=1, 3=3}
print(lruCache.get(2))  # возвращает -1 (не найдено)
print(lruCache.put(4, 4))  # удаляет ключ 1, кэш: {4=4, 3=3}
print(lruCache.get(1))  # возвращает -1 (не найдено)
print(lruCache.get(3))  # возвращает 3
print(lruCache.get(4))  # возвращает 4