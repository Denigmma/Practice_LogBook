import sys

def main():
    sys.setrecursionlimit(1 << 25)
    n, m = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(n)]
    edges = []
    for edge_id in range(m):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1
        v -= 1
        adj[u].append((v, edge_id))
        adj[v].append((u, edge_id))
        edges.append((u, v))
    time = [0]
    tin = [0] * n
    low = [0] * n
    visited = [False] * n
    is_bridge = [False] * m

    def dfs(u, parent_edge_id):
        visited[u] = True
        tin[u] = low[u] = time[0]
        time[0] += 1
        for v, edge_id in adj[u]:
            if edge_id == parent_edge_id:
                continue
            if visited[v]:
                low[u] = min(low[u], tin[v])
            else:
                dfs(v, edge_id)
                low[u] = min(low[u], low[v])
                if low[v] > tin[u]:
                    is_bridge[edge_id] = True

    for i in range(n):
        if not visited[i]:
            dfs(i, -1)

    parent = [i for i in range(n)]

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root != v_root:
            parent[v_root] = u_root

    for edge_id, (u, v) in enumerate(edges):
        if not is_bridge[edge_id]:
            union(u, v)

    component_size = dict()
    for u in range(n):
        rep = find(u)
        component_size[rep] = component_size.get(rep, 0) + 1

    result = 0
    for size in component_size.values():
        result += size * (size - 1) // 2

    bridge_edges = []
    for edge_id, (u, v) in enumerate(edges):
        if is_bridge[edge_id]:
            u_root = find(u)
            v_root = find(v)
            bridge_edges.append((u_root, v_root))

    processed_bridges = set()
    for u_root, v_root in bridge_edges:
        if (u_root, v_root) not in processed_bridges and (v_root, u_root) not in processed_bridges:
            size_u = component_size[u_root]
            size_v = component_size[v_root]
            result += size_u * size_v
            processed_bridges.add((u_root, v_root))

    print(result)

if __name__ == "__main__":
    main()

