def solve():
	import sys
	from collections import deque

	# data = sys.stdin.read().strip().split()

	# N=int(data[0])
	N=int(input())
	name2id={}
	node_type=[]
	adjaceny=[]
	in_degree=[]

	def get_node_id(name: str) -> int:
		if name not in name2id:
			idx=len(name2id)
			name2id[name]=idx
			if name.startswith("ip_") or name.startswith("op_") or name.startswith("ff_"):
				node_type.append("B")
			else:
				node_type.append("S")
			adjaceny.append([])
			in_degree.append(0)
		return name2id[name]

	idx=1
	for _ in range(N):
		# s_name=data[idx];	idx+=1
		# t_name=data[idx];	idx+=1
		# delay=float(data[idx]);	idx+=1
		s_name,t_name,delay=input().split()
		delay=float(delay)

		s_id=get_node_id(s_name)
		t_id=get_node_id(t_name)

		adjaceny[s_id].append((t_id,delay))
		in_degree[t_id]+=1
	num_nodes=len(name2id)

	import math
	disttStart=[-math.inf]*num_nodes
	distS=[-math.inf]*num_nodes
	distEnd=[-math.inf]*num_nodes

	for v in range(num_nodes):
		if node_type[v]=="B":
			disttStart[v]=0.0

	queue=deque()
	for v in range(num_nodes):
		if in_degree[v]==0:
			queue.append(v)

	while queue:
		u=queue.popleft()

		if node_type[u]=="B":
			base=disttStart[u]
			if base> -math.inf:
				for (v,cost) in adjaceny[u]:
					if node_type[v]=="B":
						if base+cost>distEnd[v]:
							distEnd[v]=base+cost
					else:
						if base+cost>distS[v]:
							distS[v]=base+cost
		else:
			base=distS[u]
			if base > -math.inf:
				for (v,cost) in adjaceny[u]:
					if node_type[v]=="B":
						if base+cost > distEnd[v]:
							distEnd[v]=base+cost
					else:
						if base + cost > distS[v]:
							distS[v]=base+cost

		for (v, _ ) in adjaceny[u]:
			in_degree[v]-=1
			if in_degree[v]==0:
				queue.append(v)

	answer=max(distEnd[v] for v in range(num_nodes) if node_type[v]=="B")
	print(f"{answer:.4f}")



solve()