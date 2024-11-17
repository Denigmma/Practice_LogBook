from collections import deque, defaultdict

n=int(input())
list_proc=[]

for i in range(n):
    m=list(map(int, input().split()))
    list_proc.append(m)

list_depend=defaultdict(list)
time=[0]*n
count_depend=[0]*n
finish_time=[0]*n
que=deque()
for i in range(n):
    data=list_proc[i]
    time[i]=data[0]
    for dep in data[1:]:
        list_depend[dep-1].append(i)
        count_depend[i]+=1
for i in range(n):
    if count_depend[i]==0:
        que.append(i)
        finish_time[i]=time[i]
while que:
    process=que.popleft()
    for dep in list_depend[process]:
        finish_time[dep]=max(finish_time[dep],finish_time[process]+time[dep])
        count_depend[dep]-=1
        if count_depend[dep]==0:
            que.append(dep)

print(max(finish_time))