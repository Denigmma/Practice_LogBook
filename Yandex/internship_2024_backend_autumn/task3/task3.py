def find_min_erase(n, list_a, list_b, list_c):
    set_a = set(list_a)
    set_b = set(list_b)
    set_c = set(list_c)
    common = set_a.intersection(set_b, set_c)
    erase_count_a = sum(1 for event in list_a if event not in common)
    erase_count_b = sum(1 for event in list_b if event not in common)
    erase_count_c = sum(1 for event in list_c if event not in common)

    return erase_count_a + erase_count_b + erase_count_c

with open('input.txt', 'r') as file:
    n = int(file.readline().strip())
    events_a = list(map(int, file.readline().strip().split()))
    events_b = list(map(int, file.readline().strip().split()))
    events_c = list(map(int, file.readline().strip().split()))

result = find_min_erase(n, events_a, events_b, events_c)
with open('output.txt', 'w') as file:
    file.write(str(result) + '\n')
