def minimal_operations(n, x, y, z, a):
    def compute_costs(k):
        costs = [(k - (ai % k)) % k for ai in a]
        return sorted(costs)[:3]
    costs_x = compute_costs(x)
    costs_y = compute_costs(y)
    costs_z = compute_costs(z)
    min_total_cost = float('inf')
    for cx in costs_x:
        for cy in costs_y:
            for cz in costs_z:
                if len({cx, cy, cz}) == 3:
                    total_cost = cx + cy + cz
                elif len({cx, cy, cz}) == 2:
                    total_cost = max(cx, cy, cz) + min(cx, cy, cz)
                else:
                    total_cost = cx
                min_total_cost = min(min_total_cost, total_cost)
    return min_total_cost

n, x, y, z = map(int, input().split())
a = list(map(int, input().split()))
print(minimal_operations(n, x, y, z, a))