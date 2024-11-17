def num_is_simple_and_kolvo_div(r):
    filter = [True] * (r + 1)
    count_div = [1] * (r + 1)
    filter[0] = filter[1] = False
    for i in range(2, r + 1):
        if filter[i]:
            for j in range(i, r + 1, i):
                filter[j] = False if j != i else True
                count = 0
                num = j
                while num % i == 0:
                    num //= i
                    count += 1
                count_div[j] *= (count + 1)
    simplenum = [x for x in range(2, r + 1) if filter[x]]
    return simplenum, count_div

def count_result(l, r):
    simplenum, count_div = num_is_simple_and_kolvo_div(r)
    simplenum_set = set(simplenum)
    result = 0
    for i in range(max(2, l), r + 1):
        if count_div[i] > 2 and count_div[i] in simplenum_set:
            result += 1
    return result

l, r = map(int, input().split())
print(count_result(l, r))

