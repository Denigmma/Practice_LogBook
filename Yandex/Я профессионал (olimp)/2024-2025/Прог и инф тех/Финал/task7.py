def max_avl_h(n):
	M=[0,1]
	while True:
		next_val=1+M[-1]+M[-2]
		if next_val>n:
			break
		M.append(next_val)
	return len(M)-1


n=int(input().strip())
print(max_avl_h(n))