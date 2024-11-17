n=int(input())
a=list(map(int, input().split()))
count=0

dict={}

for j in a:
    if j in dict:
        dict[j]+=1
    else:
        dict[j]=1

for i in dict:
    if dict[i]==1:
        count+=1
print(count)


