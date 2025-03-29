def solve():
	import sys
	data=sys.stdin.read().strip().split()
	c=int(data[0])
	n=int(data[1])
	items=list(map(int, data[2:]))
	items.sort(reverse=True)
	bins=[]
	for item in items:
		placed=False
		for i in range(len(bins)):
			if bins[i]+item<=c:
				bins[i]+=item
				placed=True
				break

		if not placed:
			bins.append(item)

	print(len(bins))

solve()