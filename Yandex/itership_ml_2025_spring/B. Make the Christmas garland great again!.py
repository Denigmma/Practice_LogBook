n = int(input())

# Построение дерева
tree = [[] for _ in range(n + 1)]
for _ in range(n - 1):
    a, b, t = map(int, input().split())
    tree[a].append((b, t))
    tree[b].append((a, t))


# Функция для нахождения самой удаленной вершины
def find_farthest(start):
    max_dist = 0
    far_node = start
    parent = [-1] * (n + 1)
    visited = [False] * (n + 1)
    stack = [(start, 0)]  # Стек для обхода в глубину
    while stack:
        current, dist = stack.pop()
        if visited[current]:
            continue
        visited[current] = True
        if dist > max_dist:
            max_dist = dist
            far_node = current
        for neighbor, t in tree[current]:
            if not visited[neighbor]:
                parent[neighbor] = current
                stack.append((neighbor, dist + t))
    return far_node, max_dist, parent


# Находим первый конец диаметра
u, _, _ = find_farthest(1)

# Находим второй конец диаметра и родительские связи
v, max_dist, parent_u = find_farthest(u)
S = max_dist  # Длина диаметра

# Восстанавливаем путь от v до u
path = []
current = v
while current != u:
    path.append(current)
    current = parent_u[current]
path.append(u)
path = path[::-1]  # Путь от u до v

# Собираем prefix_sum для вычисления расстояний
prefix_sum = [0]
current_sum = 0
current_node = u
for node in path[1:]:
    for neighbor, t in tree[current_node]:
        if neighbor == node:
            current_sum += t
            prefix_sum.append(current_sum)
            current_node = node
            break

# Находим оптимальную вершину
min_max = float('inf')
best_node = u
for i in range(len(prefix_sum)):
    current_dist = prefix_sum[i]
    other_dist = S - current_dist
    current_max = max(current_dist, other_dist)
    if current_max < min_max:
        min_max = current_max
        best_node = path[i]

print(best_node)