input = input()
intervals = input.split(',')

array=[]

def row(start, end):
    for num in range(int(start), int(end) + 1):
        array.append(num)

for i in intervals:
    if '-' in i:
        start, end = map(int, i.split('-'))
        row(start, end)
    else:
        num = int(i)
        array.append(num)
print(' '.join(map(str, array)))