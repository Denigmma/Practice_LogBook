import math
n=int(input().strip())
max_hight=math.floor(1.44*math.log2(n+2)-0.325)
print(max_hight)