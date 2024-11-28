### right solution

from collections import defaultdict, deque

MOD = 10**9 + 7

def modular_inverse(a, mod):
    return pow(a, mod - 2, mod)


def solve_gflownet(n, transitions):
    graph = defaultdict(list)
    in_degree = [0] * (n + 1)
    total_weights = [0] * (n + 1)

    for s in range(1, n + 1):
        for t, f in transitions[s]:
            graph[s].append((t, f))
            in_degree[t] += 1
            total_weights[s] += f

    topo_order = []
    queue = deque(s for s in range(1, n + 1) if in_degree[s] == 0)

    while queue:
        s = queue.popleft()
        topo_order.append(s)
        for t, _ in graph[s]:
            in_degree[t] -= 1
            if in_degree[t] == 0:
                queue.append(t)

    probabilities = [0] * (n + 1)
    probabilities[1] = 1

    for s in topo_order:
        total_f = total_weights[s] % MOD
        if total_f == 0:
            continue

        total_f_inv = modular_inverse(total_f, MOD)

        for t, f in graph[s]:
            probabilities[t] += probabilities[s] * f % MOD * total_f_inv % MOD
            probabilities[t] %= MOD

    final_states = [s for s in range(1, n + 1) if len(transitions[s]) == 0]

    result = []
    for s in sorted(final_states):
        result.append((s, probabilities[s]))
    return result


n = int(input())
transitions = defaultdict(list)
for i in range(1, n + 1):
    line = list(map(int, input().split()))
    k = line[0]
    for j in range(k):
        s_prime, f = line[2 * j + 1], line[2 * j + 2]
        transitions[i].append((s_prime, f))

result = solve_gflownet(n, transitions)

for s, prob in result:
    print(s, prob)
