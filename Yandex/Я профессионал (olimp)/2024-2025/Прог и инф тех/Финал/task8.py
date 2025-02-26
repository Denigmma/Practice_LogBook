def countDigitAll(d,n):
	if n<0:
		return 0
	count=0
	position=1
	while position<=n:
		left=n//(position*10)
		digit=(n//position)%10
		right=n%position
		if digit > d:
			count += (left + 1) * position
		elif digit == d:
			count += left * position + right + 1
		else:
			count += left * position
		if d == 0:
			count -= position
		position *= 10
	return count

n=int(input())
freq=[0]*10

for d in range(10):
	freq[d]=countDigitAll(d,n)

print(*freq)