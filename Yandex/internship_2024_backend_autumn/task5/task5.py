def longest_common_prefix_length(word1, word2):
    length = 0
    while length < min(len(word1), len(word2)) and word1[length] == word2[length]:
        length += 1
    return length


def process_queries(dictionary, queries):
    results = []

    for query in queries:
        comparisons = 0
        total_prefix_length = 0

        for word in dictionary:
            comparisons += 1
            lcp_length = longest_common_prefix_length(word, query)
            total_prefix_length += lcp_length

            if word == query:
                break

        total_actions = comparisons + total_prefix_length
        results.append(total_actions)

    return results


# Чтение данных из файла
with open('input.txt', 'r') as file:
    n = int(file.readline().strip())
    dictionary = [file.readline().strip() for _ in range(n)]
    q = int(file.readline().strip())
    queries = [file.readline().strip() for _ in range(q)]

# Обрабатываем запросы
results = process_queries(dictionary, queries)

# Запись результатов в файл
with open('output.txt', 'w') as file:
    for result in results:
        file.write(str(result) + '\n')
