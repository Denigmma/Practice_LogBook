from math import log2
n = int(input())
amounts = [int(input()) for _ in range(n)]
results = []
def max_possible_money(amount, depth, last=0):
    if depth == 0:
        return 0
    if amount < 2 ** depth - 1:
        return -1
    pw = 2 ** int(log2(amount))
    p = max_possible_money(amount - pw, depth - 1, pw)
    while p < 0 or pw == last:
        pw //= 2
        p = max_possible_money(amount - pw, depth - 1, pw)
    return pw + p
for i in range(n):
    results.append(max_possible_money(amounts[i], 3))
print('\n'.join(map(str, results)))



# import bisect
# import sys
#
# def solve():
#     n = int(sys.stdin.readline().strip())
#     a = [int(sys.stdin.readline().strip()) for _ in range(n)]
#     powers = [1 << i for i in range(60)]
#     S = []
#     for i in range(60):
#         for j in range(i + 1, 60):
#             for k in range(j + 1, 60):
#                 S.append(powers[i] + powers[j] + powers[k])
#     S.sort()
#     results = []
#     for x in a:
#         idx = bisect.bisect_right(S, x) - 1
#         results.append(str(S[idx]) if idx >= 0 else "-1")
#     sys.stdout.write("\n".join(results) + "\n")
# solve()