def min_changes_to_like_schedule(n, m, a):
    left, right = 0, 10**9
    def can_make_m_good_days(k):
        changes = []
        for i in range(2, n):
            if a[0] <= a[i] <= a[1]:
                changes.append(0)
            else:
                if a[i] < a[0]:
                    changes.append(a[0] - a[i])
                else:
                    changes.append(a[i] - a[1])
        changes.sort()
        return sum(changes[:m]) <= k
    while left < right:
        mid = (left + right) // 2
        if can_make_m_good_days(mid):
            right = mid
        else:
            left = mid + 1

    return left

n, m = map(int, input().split())
a = list(map(int, input().split()))
print(min_changes_to_like_schedule(n, m, a))