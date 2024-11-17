from collections import Counter

def hfp(series, need_chars, maxlen):
    need_count = Counter(need_chars)
    need_unique_chars = len(need_count)
    currentcount = Counter()
    current_unique_chars = 0
    l=0
    n=len(series)
    rightpass = ""
    for r in range(n):
        char = series[r]
        currentcount[char] += 1
        if char in need_count and currentcount[char] == need_count[char]:
            current_unique_chars += 1
        while r - l + 1 > maxlen:
            left_char = series[l]
            if left_char in need_count and currentcount[left_char] == need_count[left_char]:
                current_unique_chars -= 1
            currentcount[left_char] -= 1
            l += 1
        if current_unique_chars == need_unique_chars:
            currentpass = series[l:r + 1]
            if len(currentpass) > len(rightpass) or (
                    len(currentpass) == len(rightpass) and l > series.find(rightpass)):
                rightpass = currentpass
    return rightpass if rightpass else "-1"

series = input().strip()
need_chars = input().strip()
maxlen = int(input().strip())

res = hfp(series, need_chars, maxlen)
print(res)
