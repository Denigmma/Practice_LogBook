n=int(input())
x=list(map(int,input().split()))

for first in range(97,123):
	a=[0]*n
	a[0]=first
	for i in range(n-1):
		a[i+1]=a[i]^x[i]
	if a[n-1]^a[0]!=x[n-1]:
		continue
	if all(97<=v<=122 for v in a):
		print("".join(chr(v) for v in a))
		break