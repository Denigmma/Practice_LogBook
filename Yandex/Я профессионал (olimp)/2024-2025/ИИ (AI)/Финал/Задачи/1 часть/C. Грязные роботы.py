def solve():
	import sys
	line=sys.stdin.readline().strip()
	if not line:
		return
	parts=line.split()
	N=int(parts[0])
	T=int(parts[1])
	other_state=list(map(int,parts[2:2+(N-1)]))

	d=sum(other_state)

	observ=[]
	for _ in range(T):
		line=sys.stdin.readline().strip()
		observ.append(line)

	daterm=False
	my_state=None
	for t in range(T):
		if not daterm:
			if d==0:
				if t==0:
					daterm=True
					my_state=1
			else:
				if t==(d-1):
					daterm=True
					my_state=0
				elif t==d:
					daterm=True
					my_state=1
		if daterm:
			print(my_state)
		else:
			print(-1)

solve()