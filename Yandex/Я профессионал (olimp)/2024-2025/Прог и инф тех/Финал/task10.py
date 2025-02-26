import heapq

def z_function(s):
	n = len(s)
	z = [0] * n
	l = 0
	r = 0
	for i in range(1, n):
		if i < r:
			z[i] = min(r - i, z[i - l])
		while i + z[i] < n and s[z[i]] == s[i + z[i]]:
			z[i] += 1
		if i + z[i] > r:
			l = i
			r = i + z[i]
	return z

a=input()
b=input()
s=input()

n_a,n_b,n_s=len(a),len(b),len(s)

za=z_function(a+'#'+s)
zb=z_function(b+'#'+s)

oa=n_a+1
ob=n_b+1

La=[0]*n_s
for i in range(n_s):
	m=zb[oa+i]
	if m>n_a:
		m=n_a
	La[i]=m

Lb=[0]*n_s
for i in range(n_s):
	m=zb[ob+i]
	if m>n_b:
		m=n_b
	Lb[i]=m

ivB=[]
for j in range(n_s):
	if Lb[j]>0:
		ivB.append((j, j+Lb[j], Lb[j]))
ivB.sort(key=lambda x: x[0])

h=[]
ans=0
p=0
del_map={}

def pop_lazy():
	while h:
		w,e,st=h[0]
		if del_map.get((e,st,-w),0)>0:
			del_map[(e,st,-w)]-=1
			heapq.heappop(h)
		else:
			break

for i in range(n_s):
	la=La[i]
	if la==0:
		pop_lazy()
		while h and h[0][1]<=i:
			heapq.heappop(h)
		continue
	ea=i+la
	while p<len(ivB) and ivB[p][0]<ea:
		st_b,en_b,wb=ivB[p]
		p+=1
		if en_b>i:
			heapq.heappush(h,(-wb,en_b,st_b))
	pop_lazy()
	while h and h[0][1]<=i:
		heapq.heappop(h)
	pop_lazy()
	if h:
		ans=max(ans,la-h[0][0])

print(ans)


