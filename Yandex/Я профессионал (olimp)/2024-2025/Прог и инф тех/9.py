MOD = 10**9 + 7

def solve(s):
    n = len(s)
    num_ones = s.count('1')
    runs = []
    i = 0
    while i < n:
        if s[i] == '0':
            run_length = 0
            while i < n and s[i] == '0':
                run_length += 1
                i += 1
            runs.append(run_length)
        else:
            i += 1
    total_ways_to_select_zeros = 1
    for run_length in runs:
        total_ways_to_select_zeros = total_ways_to_select_zeros * (run_length + 1) % MOD
    total_subseq_without_adjacent_zeros = total_ways_to_select_zeros * pow(2, num_ones, MOD) % MOD
    total_subseq = pow(2, n, MOD)
    number_of_interesting_subseq = (total_subseq - total_subseq_without_adjacent_zeros + MOD) % MOD
    return number_of_interesting_subseq

C = int(input())
for _ in range(C):
    s = input().strip()
    print(solve(s))