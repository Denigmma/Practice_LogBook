'''
Реализовать класс RandomizedSet:

RandomizedSet() Инициализирует RandomizedSet объект.
bool insert(int val) Вставляет элемент val в набор, если его там нет. Возвращает true если элемента не было, false в противном случае.
bool remove(int val) Удаляет элемент val из набора, если он присутствует. Возвращает true если элемент присутствовал, false в противном случае.
int getRandom() Возвращает случайный элемент из текущего набора элементов (гарантируется, что при вызове этого метода существует хотя бы один элемент). Каждый элемент должен иметь одинаковую вероятность быть возвращённым.
Вы должны реализовать функции класса таким образом, чтобы каждая функция работала со средней O(1) временной сложностью.

Example 1:

Input
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
Output
[null, true, false, true, 2, true, false, 2]

Объяснение
RandomizedSet randomizedSet = new RandomizedSet();
randomizedSet.insert(1); // Вставляет 1 в набор. Возвращает true, так как 1 был успешно вставлен.
randomizedSet.remove(2); // Возвращает false, так как 2 не существует в наборе.
randomizedSet.insert(2); // Вставляет 2 в набор, возвращает true. Теперь набор содержит [1,2].
randomizedSet.getRandom(); // getRandom() должен возвращать 1 или 2 случайным образом.
randomizedSet.remove(1); // Удаляет 1 из набора, возвращает true. Теперь набор содержит [2].
randomizedSet.insert(2); // 2 уже было в наборе, поэтому возвращается false.
randomizedSet.getRandom(); // Поскольку 2 — единственное число в наборе, getRandom() всегда будет возвращать 2.


Ограничения:

-231 <= val <= 231 - 1
Не более 2 * 105 звонков будет сделано на insert, remove и getRandom.
При вызове в структуре данных будет по крайней мере один getRandom элемент.
'''
import random

class RandomizedSet:

    def __init__(self):
        # ключ — сам элемент, значение — его индекс в списке
        self.dict={}
        self.list=[]

    def insert(self, val: int) -> bool:
        '''
        если val нет в словаре → добавляем в конец списка, сохраняем индекс в словарь
        иначе — возвращаем False
        '''
        if val in self.dict:
            return False
        self.dict[val]=len(self.list)
        self.list.append(val)
        return True

    def remove(self, val: int) -> bool:
        '''
        меняем местами удаляемый элемент с последним в списке,
        удаляем последний, и обновляем индексы в словаре
        '''
        if val not in self.dict:
            return False
        idx_to_remove = self.dict[val]
        last_element = self.list[-1]

        self.list[idx_to_remove], self.list[-1] = self.list[-1], self.list[idx_to_remove]
        self.dict[last_element] = idx_to_remove

        self.list.pop()
        del self.dict[val]
        return True

    def getRandom(self) -> int:
        '''
        используем random.choice() по списку
        '''
        return random.choice(self.list)


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()


results = []

# 1. создаём объект
obj = RandomizedSet()
results.append(None)  # потому что __init__ ничего не возвращает

# 2. вызываем insert(1)
results.append(obj.insert(1))  # True

# 3. вызываем remove(2)
results.append(obj.remove(2))  # False

# 4. вызываем insert(2)
results.append(obj.insert(2))  # True

# 5. вызываем getRandom()
results.append(obj.getRandom())  # может вернуть 1 или 2

# 6. вызываем remove(1)
results.append(obj.remove(1))  # True

# 7. вызываем insert(2)
results.append(obj.insert(2))  # False (2 уже есть)

# 8. вызываем getRandom()
results.append(obj.getRandom())  # вернёт 2 (потому что остался только 2)

print(results)