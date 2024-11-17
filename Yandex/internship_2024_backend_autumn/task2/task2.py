with open('input.txt', 'r') as infile:
    N, B = map(int, infile.readline().strip().split())
    A = list(map(int, infile.readline().strip().split()))

pos = A.index(B)
more, less = 0, 0
left_sums = {}
right_sums = {}

for i in range(pos - 1, -1, -1):
    if A[i] < B:
        less += 1
    else:
        more += 1
    balance = more - less
    left_sums[balance] = left_sums.get(balance, 0) + 1

more, less = 0, 0
for i in range(pos + 1, N):
    if A[i] < B:
        less += 1
    else:
        more += 1
    balance = more - less
    right_sums[balance] = right_sums.get(balance, 0) + 1

result_count = left_sums.get(0, 0) + right_sums.get(0, 0) + 1

for balance in left_sums:
    if -balance in right_sums:
        result_count += left_sums[balance] * right_sums[-balance]

with open('output.txt', 'w') as outfile:
    outfile.write(str(result_count))
