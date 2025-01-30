def count_cuts(n, s, a):
    total_sum = 0
    for l in range(n):
        r = l
        current_sum = 0
        cuts = 0
        while r < n:
            current_sum += a[r]
            if current_sum > s:
                cuts += 1
                current_sum = a[r]
            r += 1
            total_sum += cuts + 1
    return total_sum

n, s = map(int, input().split())
a = list(map(int, input().split()))
print(count_cuts(n, s, a))
