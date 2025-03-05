n=int(input())
list=list(map(int, input().split()))

last_pos = {}
min_distance = float('inf')

for i in range(n):
    num = list[i]
    if num in last_pos:
        current_dist = i - last_pos[num]
        if current_dist < min_distance:
            min_distance = current_dist
        last_pos[num] = i
    else:
        last_pos[num] = i

if min_distance == float('inf'):
    print(-1)
else:
    print(min_distance - 1)
