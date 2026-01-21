# -*- coding: utf-8 -*-

import sys

DIGITS36 = "0123456789abcdefghijklmnopqrstuvwxyz"

ONES = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
TENS = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
HUNDS = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]

VAL = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
ALLOWED_SUB = {("I", "V"), ("I", "X"), ("X", "L"), ("X", "C"), ("C", "D"), ("C", "M")}


def int_to_roman(n: int) -> str:
    m = n // 1000
    n %= 1000
    return ("M" * m) + HUNDS[n // 100] + TENS[(n // 10) % 10] + ONES[n % 10]


def roman_to_int_strict(s: str):
    if not s or s == "0":
        return None
    for ch in s:
        if ch not in VAL:
            return None

    total = 0
    i = 0
    n = len(s)

    while i < n:
        v = VAL[s[i]]
        if i + 1 < n:
            v2 = VAL[s[i + 1]]
            if v < v2:
                if (s[i], s[i + 1]) not in ALLOWED_SUB:
                    return None
                total += (v2 - v)
                i += 2
                continue
        total += v
        i += 1

    if total <= 0:
        return None

    if int_to_roman(total) != s:
        return None

    return total


def to_base36(x: int) -> str:
    if x == 0:
        return "0"
    out = []
    while x:
        x, r = divmod(x, 36)
        out.append(DIGITS36[r])
    out.reverse()
    return "".join(out)


def main():
    data = sys.stdin.buffer

    max1 = max2 = max3 = None

    while True:
        line = data.readline()
        if not line:
            print("0")
            return

        s = line.strip().decode("ascii", "ignore")
        if s == "":
            print("0")
            return

        if s == "0":
            break

        v = roman_to_int_strict(s)
        if v is None:
            print("0")
            return

        if v == max1 or v == max2 or v == max3:
            continue

        if max1 is None or v > max1:
            max3 = max2
            max2 = max1
            max1 = v
        elif max2 is None or v > max2:
            max3 = max2
            max2 = v
        elif max3 is None or v > max3:
            max3 = v

    extra = data.readline()
    if extra:
        print("0")
        return

    if max1 is None or max2 is None or max3 is None:
        print("0")
        return

    print(to_base36(max1 * max2 * max3))


if __name__ == "__main__":
    main()
