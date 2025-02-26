import heapq

def huffman_codes(leaf_count):
    # минимальная куча
    heap = [[weight, [symbol, ""]] for symbol, weight in enumerate([1] * leaf_count)]
    heapq.heapify(heap)

    # дерево
    while len(heap) > 1:
        low = heapq.heappop(heap)
        high = heapq.heappop(heap)

        for pair in low[1:]:
            pair[1] = '0' + pair[1]
        for pair in high[1:]:
            pair[1] = '1' + pair[1]

        heapq.heappush(heap, [low[0] + high[0]] + low[1:] + high[1:])

    codes = [symbol[1] for symbol in heap[0][1:]]
    return codes


# проверка условия Фано
def check_fano_condition(codes):
    for i in range(len(codes)):
        for j in range(len(codes)):
            if i != j and codes[j].startswith(codes[i]):
                return False
    return True


# суммы длин кодов
def calculate_sum_of_lengths(codes):
    return sum(len(code) for code in codes)


leaf_count = 100
codes = huffman_codes(leaf_count)

if check_fano_condition(codes):
    print(codes)
    print(f"Сумма длин кодов: {calculate_sum_of_lengths(codes)}")
else:
    print("Условие Фано нарушено.")
