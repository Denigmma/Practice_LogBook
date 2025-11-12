import sys
import math
import bisect

sys.setrecursionlimit(1 << 20)

def read_all_tokens():
    return sys.stdin.read().strip().split()

def solve_one(n, edges, points):
    g = [[] for _ in range(n)]
    for u, v in edges:
        u -= 1; v -= 1
        g[u].append(v); g[v].append(u)

    root = 0

    parent = [-1] * n
    parent[root] = root
    order = [root]
    for v in order:
        for to in g[v]:
            if to == parent[v]:
                continue
            parent[to] = v
            order.append(to)

    sz = [1] * n
    for v in reversed(order):
        for to in g[v]:
            if to == parent[v]:
                continue
            sz[v] += sz[to]

    children = [[] for _ in range(n)]
    for v in range(n):
        for to in g[v]:
            if parent[to] == v:
                children[v].append(to)


    root_point = min(range(n), key=lambda i: (points[i][1], points[i][0]))

    ans = [-1] * n

    def assign(u, pts_list, anchor_idx, parent_point_idx=None):
        """u — вершина, pts_list — индексы точек её поддерева,
           anchor_idx — точка для u, parent_point_idx — точка родителя (None для корня)"""
        ans[u] = anchor_idx
        if not children[u]:
            return

        ax, ay = points[anchor_idx]

        others = [p for p in pts_list if p != anchor_idx]
        angs = []
        for j in others:
            angs.append((math.atan2(points[j][1] - ay, points[j][0] - ax), j))
        angs.sort()

        if parent_point_idx is not None:
            apar = math.atan2(points[parent_point_idx][1] - ay,
                              points[parent_point_idx][0] - ax)
            arr = [a for a, _ in angs]
            i0 = bisect.bisect(arr, apar)
            angs = angs[i0:] + angs[:i0]

        others_sorted = [j for _, j in angs]

        ptr = 0
        for v in children[u]:
            k = sz[v]
            block = others_sorted[ptr:ptr + k]
            ptr += k
            child_anchor = block[0]
            assign(v, block, child_anchor, anchor_idx)

    assign(root, list(range(n)), root_point, None)

    return [x + 1 for x in ans]

def main():
    tok = read_all_tokens()
    it = iter(tok)
    t = int(next(it))
    out_lines = []
    for _ in range(t):
        n = int(next(it))
        edges = [(int(next(it)), int(next(it))) for _ in range(n - 1)]
        points = [(float(next(it)), float(next(it))) for _ in range(n)]
        perm = solve_one(n, edges, points)
        out_lines.append(" ".join(map(str, perm)))
    print("\n".join(out_lines))

if __name__ == "__main__":
    main()
