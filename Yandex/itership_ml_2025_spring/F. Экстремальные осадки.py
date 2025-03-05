from collections import deque


def main():
    n = int(input())
    t1, t2, k = map(int, input().split())
    matrix = []
    for _ in range(n):
        row = list(map(int, input().split()))
        matrix.append(row)

    visited = [[False for _ in range(n)] for _ in range(n)]

    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    for i in range(n):
        for j in range(n):
            if not visited[i][j] and matrix[i][j] >= t1:
                queue = deque()
                queue.append((i, j))
                visited[i][j] = True
                area = 1
                max_val = matrix[i][j]
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < n and 0 <= ny < n:
                            if not visited[nx][ny] and matrix[nx][ny] >= t1:
                                visited[nx][ny] = True
                                queue.append((nx, ny))
                                area += 1
                                if matrix[nx][ny] > max_val:
                                    max_val = matrix[nx][ny]
                if area > k:
                    print(False)
                    return
                if max_val < t2:
                    print(False)
                    return
    print(True)


main()