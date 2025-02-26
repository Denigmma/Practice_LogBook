def calculate_b_weight(x, b):
    weight = 0
    while x > 0:
        digit = x % b
        weight += digit * digit
        x //= b
    return weight

def solve():
    n, b = map(int, input().split())
    arr = list(map(int, input().split()))
    weighted_arr = [(calculate_b_weight(x, b), i, x) for i, x in enumerate(arr)]
    weighted_arr.sort(key=lambda x: x[0])
    print(" ".join(str(x[2]) for x in weighted_arr))

solve()