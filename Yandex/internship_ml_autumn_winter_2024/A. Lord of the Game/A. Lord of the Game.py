def max_res(n, stones):
    min_stones = min(stones)
    max_stones = max(stones)

    first_min = -1
    last_min = -1
    first_max = -1
    last_max = -1

    for i in range(n):
        if stones[i] == min_stones:
            first_min = i
            break

    for i in range(n - 1, -1, -1):
        if stones[i] == min_stones:
            last_min = i
            break

    for i in range(n):
        if stones[i] == max_stones:
            first_max = i
            break

    for i in range(n - 1, -1, -1):
        if stones[i] == max_stones:
            last_max = i
            break
    sum1 = sum(stones[first_min:last_max + 1])

    sum2 = sum(stones[first_max:last_min + 1])

    print(max(sum1, sum2))

n=int(input())
stones=list(map(int,input().split()))
max_res(n,stones)




# def max_res(n,arr):
#     min_stack=min(arr)
#     max_stack=max(arr)
#
#     i_min = []
#     for i in range(n):
#         if min_stack==arr[i]:
#             i_min.append(i)
#
#     i_max = []
#     for i in range(n):
#         if max_stack==arr[i]:
#             i_max.append(i)
#
#     s = 0
#     for i in i_min:
#         for j in i_max:
#             left, right = min(i, j), max(i, j)
#             s = max(s, sum(arr[left:right+1]))
#     print(s)
#
# n=int(input())
# arr=list(map(int,input().split()))
# max_res(n,arr)