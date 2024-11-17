n = int(input().strip())
data = list(map(int, input().strip().split()))

flag = True
difference = []
previous_value = 0

for value in data:
    if value == -1:
        previous_value += 1
        difference.append(previous_value - (previous_value - 1))
    else:
        if value < previous_value:
            flag = False
            break
        difference.append(value - previous_value)
        previous_value = value

if flag:
    print("YES")
    print(" ".join(map(str, difference)))
else:
    print("NO")
