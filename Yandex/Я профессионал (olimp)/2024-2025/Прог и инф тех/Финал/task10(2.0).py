import math

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

def build_segment_tree(arr):
	n=len(arr)
	size=1
	while size<n:
		size*=2
	seg=[0]*(2*size)
	for i in range(n):
		seg[size+i]=arr[i]
	for i in range(size-1,0,-1):
		seg[i]=max(seg[2*i],seg[2*i+1])
	return seg, size

def seg_query(seg,size,l,r):
	res=0
	l+=size
	r+=size
	while l<=r:
		if l%2==1:
			res=max(res, seg[l])
			l+=1
		if r%2==0:
			res=max(res,seg[r])
			r-=1
		l//=2
		r//=2
	return res

a=input().strip()
b=input().strip()
s=input().strip()

n=len(s)
na=len(a)
nb=len(b)

sa=a+'#'+s
za=z_function(sa)
offset_a=len(a)+1
L_a=[min(za[offset_a+i],na) for i in range(n)]

sb=b+'#'+s
zb=z_function(sb)
offset_b=len(b)+1
L_b=[min(zb[offset_b+i],nb) for i in range(n)]

seg,size=build_segment_tree(L_b)

ans=0
for i in range(n):
	if L_a[i]>0:
		j_low=i
		j_hight=min(n-1,i+L_a[i]-1)
		max_Lb=seg_query(seg,size,j_low,j_hight)
		ans=max(ans,L_a[i]+max_Lb)

print(ans)