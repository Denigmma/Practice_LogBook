# def replace_vars(arr):
#     alphabet = "ABCDEFGH"
#     original_n = len(arr)
#
#     i = 0
#     # Проходим по «головной» части массива — до тех пор, пока i < original_n
#     while i < original_n:
#         # Если текущий элемент — строка, а за ней хотя бы один int
#         if isinstance(arr[i], str) and i + 1 < len(arr) and isinstance(arr[i + 1], int):
#             # Определяем конец «ключа» (имя + все подряд идущие числа)
#             j = i + 1
#             while j < len(arr) and isinstance(arr[j], int):
#                 j += 1
#             # Теперь arr[i:j] — это наш ключ
#             key = arr[i:j]  # например ['x', 5, 6] или ['x',1,2,3]
#
#             # Ищем его в хвосте (там же лежат все ранее добавленные [имя, ...числа..., символ])
#             found = False
#             tail_idx = original_n
#             while tail_idx < len(arr):
#                 # читаем имя
#                 rec_j = tail_idx + 1
#                 # читаем подряд числа
#                 while rec_j < len(arr) and isinstance(arr[rec_j], int):
#                     rec_j += 1
#                 # теперь arr[tail_idx:rec_j] — ключ записи, а arr[rec_j] — её символ
#                 if arr[tail_idx:rec_j] == key:
#                     symbol = arr[rec_j]
#                     found = True
#                     break
#                 # прыгаем на следующую запись
#                 tail_idx = rec_j + 1
#
#             if not found:
#                 # новый маппинг: следующая буква
#                 # буква по порядку — просто длина уже записанных в хвосте записей
#                 mapped_count = 0
#                 scan = original_n
#                 while scan < len(arr):
#                     mapped_count += 1
#                     # перепрыгнуть через запись
#                     k = scan + 1
#                     while k < len(arr) and isinstance(arr[k], int):
#                         k += 1
#                     scan = k + 1
#                 symbol = alphabet[mapped_count]
#                 # добавляем [имя, ...числа..., символ] в хвост
#                 arr.extend(key + [symbol])
#
#             # распространяем замену в «голове»:
#             arr[i] = symbol
#             # удаляем старые числа
#             del arr[i + 1:j]
#             # сужаем original_n на длину удалённых чисел
#             original_n -= (j - (i + 1))
#             # ступаем дальше
#             i += 1
#         else:
#             i += 1
#
#     # Всё — обрезаем хвост с записями маппинга
#     arr[:] = arr[:original_n]
#     return arr
#
#
# # Проверка:
# array = ["x", 1,"or", "x", 1, 2, 3, "or", "x", 5, 6, "and", "x", 1, 2, 3, "and", "x",1]
# print(replace_vars(array))



# ПЕРЕПИСАЛ КАК ДЛЯ МТ 1
# def replace_vars_tm(tape):
#     """
#     Симулируем Turing‑подобную машину:
#       • tape      — список «ячее́к» (каждая ячейка хранит по одному элементу),
#       • head      — текущая позиция «головки»,
#       • оригинал  — граница между входными данными и областью маппингов (она растёт по мере добавления новых записей),
#       • мы читаем/пишем всегда в tape[head], двигаем head += ±1, и храним «состояние» через обычные if‑ветвления.
#     """
#     alphabet = "abcdefghijklmnopqrstuvwxyz"
#     # запомним первичную границу tape: до неё — входная «лента»
#     original_end = len(tape)
#
#     head = 0          # позиция головки
#     mapped_count = 0  # сколько уже создано маппингов
#     # состояние: ищем переменную на ленте (search), затем ищем/дописываем маппинг (map), затем пишем символ и «стираем» числа (write), и снова search
#     state = 'search'
#
#     # вспомогательные переменные для границ найденного ключа
#     key_start = key_end = None
#     found_symbol = None
#
#     while state != 'halt':
#         if state == 'search':
#             # если в «входной» области на head встречаем string + int
#             if head + 1 < original_end and isinstance(tape[head], str) and isinstance(tape[head+1], int):
#                 key_start = head
#                 # идём вправо, пока на ленте int
#                 head += 1
#                 while head < original_end and isinstance(tape[head], int):
#                     head += 1
#                 key_end = head      # позиция сразу после последнего числа
#                 # переходим к поиску в области маппингов
#                 map_pos = original_end
#                 state = 'map'
#             else:
#                 # иначе просто шагаем по ленте
#                 head += 1
#                 # если вышли за «изначальный» конец — всё сделано
#                 if head >= original_end:
#                     state = 'halt'
#
#         elif state == 'map':
#             # ищем в области маппингов запись, совпадающую с tape[key_start:key_end]
#             found = False
#             while map_pos < len(tape):
#                 # считываем длину ключа в этой записи
#                 # сначала имя
#                 rec_head = map_pos + 1
#                 while rec_head < len(tape) and isinstance(tape[rec_head], int):
#                     rec_head += 1
#                 # rec_head — позиция символа‑маппера
#                 rec_key_end = rec_head
#                 rec_symbol_pos = rec_head
#                 # сравним длину и содержимое ключа
#                 length = key_end - key_start
#                 if tap_slice := tape[map_pos:map_pos+length] == tape[key_start:key_end]:
#                     # нашли
#                     found_symbol = tape[rec_symbol_pos]
#                     found = True
#                     break
#                 # иначе прыгаем к следующей записи
#                 map_pos = rec_symbol_pos + 1
#
#             if not found:
#                 # создаём новый маппинг: букву из alphabet
#                 found_symbol = alphabet[mapped_count]
#                 # «записываем» ключ и символ на хвост ленты ячейка за ячейкой
#                 for i in range(key_start, key_end):
#                     tape.append(tape[i])
#                 tape.append(found_symbol)
#                 mapped_count += 1
#
#             # после поиска/дописывания маппинга переходим к «стиранию» чисел в исходной области
#             # сначала возвращаем голову в начало ключа
#             head = key_start
#             state = 'write'
#
#         elif state == 'write':
#             # пишем symbol вместо имени+чисел
#             tape[head] = found_symbol
#             # «стираем» те числа, что шли после: сдвигаем весь хвост ленты влево на (key_end-key_start-1) позиций
#             to_delete = (key_end - key_start) - 1
#             for _ in range(to_delete):
#                 # смещаем все ячейки от head+1 до конца на одну позицию влево
#                 for k in range(head+1, len(tape)-1):
#                     tape[k] = tape[k+1]
#                 tape.pop()  # и укорачиваем ленту на пустую позицию справа
#             # сдвигаем «границу» оригинальной области
#             original_end -= to_delete
#             # переходим дальше — сразу после записанного символа
#             head = key_start + 1
#             state = 'search'
#
#     # в итоге tape[0:original_end] — ваш результат
#     return tape[:original_end]
# # Пример:
# array = ["x",1,2,3,"or","x",5,6,"and","x",1,2,3]
# print(replace_vars_tm(array))
# # → ['a', 'or', 'b', 'and', 'a']

def replace_vars_tm(tape):
    """
    Симуляция однолинейной машины Тьюринга без циклов:
      – tape      — лента (список) ячеек;
      – head      — позиция головки, двигается только ±1;
      – original_end — граница входной области (растёт append’ами);
      – состояния (search, collect, map, write, halt) оформлены рекурсивно.
    """
    alphabet     = "abcdefghijklmnopqrstuvwxyz"
    original_end = len(tape)
    head         = 0
    mapped_count = 0
    key_start = key_end = None
    found_symbol = None
    map_pos      = None

    # ——— движения головки ———
    def move_right():
        nonlocal head
        head += 1

    # ——— поиск следующей переменной ———
    def state_search():
        nonlocal head
        if head >= original_end:
            return  # halt
        if isinstance(tape[head], str) and head+1 < original_end and isinstance(tape[head+1], int):
            nonlocal key_start
            key_start = head
            return state_collect()
        move_right()
        return state_search()

    # ——— собираем все цифры после имени ———
    def state_collect():
        nonlocal head, key_end
        if head+1 < original_end and isinstance(tape[head+1], int):
            move_right()
            return state_collect()
        key_end = head + 1
        return state_map_scan()

    # ——— сканируем маппинги в хвосте ———
    def state_map_scan():
        nonlocal map_pos
        map_pos = original_end
        return scan_one_mapping()

    def scan_one_mapping():
        nonlocal map_pos, found_symbol, mapped_count
        # если ушли за конец ленты — заведём новый маппинг
        if map_pos >= len(tape):
            found_symbol = alphabet[mapped_count]
            create_mapping()
            return state_write()

        # где кончается ключ в этой записи
        rec_key_end = skip_ints(map_pos+1)
        rec_key_len = rec_key_end - map_pos
        my_key_len  = key_end   - key_start

        # только если длины равны и сами элементы совпадают — считаем, что нашли
        if rec_key_len == my_key_len and tape[map_pos:rec_key_end] == tape[key_start:key_end]:
            found_symbol = tape[rec_key_end]
            return state_write()

        # иначе — прыгаем к следующей записи
        map_pos = rec_key_end + 1
        return scan_one_mapping()


    def skip_ints(pos):
        if pos < len(tape) and isinstance(tape[pos], int):
            return skip_ints(pos+1)
        return pos

    # ——— создаём новый маппинг в конце ленты ———
    def create_mapping():
        nonlocal mapped_count
        for i in range(key_start, key_end):
            tape.append(tape[i])
        tape.append(found_symbol)
        mapped_count += 1

    # ——— записываем символ и «стираем» числа по одной ячейке ———
    def state_write():
        nonlocal head, original_end
        # шаг 1: встаём в начало ключа
        head = key_start
        # шаг 2: пишем найденный символ
        tape[head] = found_symbol
        # шаг 3: удаляем числа, сдвигая ленту слева направо, начиная с head+1
        to_delete = (key_end - key_start) - 1

        def shift_left_from(pos):
            # переносим каждый элемент вправо от pos на одну ячейку влево
            if pos < len(tape) - 1:
                tape[pos] = tape[pos+1]
                shift_left_from(pos+1)

        def delete_n(n):
            if n > 0:
                shift_left_from(key_start + 1)
                tape.pop()
                delete_n(n-1)

        delete_n(to_delete)

        # шаг 4: корректируем границу входной области
        original_end -= to_delete
        # шаг 5: переходим за записанный символ
        head = key_start + 1
        return state_search()

    # ——— стартуем машину ———
    state_search()
    # возвращаем только обработанную часть ленты
    return tape[:original_end]


# Пример:
array = ["x",1,2,3,"or","x",5,6,"and","x",1,2,3, "x",1,"x",4]
print(replace_vars_tm(array))
# → ['a', 'or', 'b', 'and', 'a']

