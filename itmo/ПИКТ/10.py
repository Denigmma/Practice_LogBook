import sys

def main():
    data = sys.stdin.read().strip().splitlines()
    if not data:
        return

    S, N = map(int, data[0].split())
    carry = 0
    res_digits = []

    for line in data[1:]:
        parts = line.split()
        if not parts:
            continue
        digits = list(map(int, parts))
        if len(digits) == N and all(d == 0 for d in digits):
            break

        total = carry + sum(digits)
        res_digits.append(total % S)
        carry = total // S

    while carry > 0:
        res_digits.append(carry % S)
        carry //= S

    if not res_digits:
        print("0")
        return

    print("".join(str(d) for d in reversed(res_digits)))

if __name__ == "__main__":
    main()
