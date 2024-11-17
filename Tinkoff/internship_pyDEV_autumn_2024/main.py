from collections import defaultdict

def t_in_sec(t_str):
    h, m, s = map(int, t_str.split(":"))
    return h * 3600 + m * 60 + s

start = input().strip()
n = int(input().strip())
data = defaultdict(lambda: {"s": {}, "cnt": 0, "pen": 0})
start_sec = t_in_sec(start)

for i in range(n):
    z = input().strip().split()
    t_name = z[0].strip('"')
    q_time = z[1]
    srv_id = z[2]
    res = z[3]
    q_sec = t_in_sec(q_time)
    if res == "PONG":
        continue

    if srv_id not in data[t_name]["s"]:
        data[t_name]["s"][srv_id] = {"fail": 0, "succ": False}

    srv = data[t_name]["s"][srv_id]

    if res == "DENIED" or res == "FORBIDEN":
        srv["fail"] += 1
    elif res == "ACCESSED" and not srv["succ"]:
        srv["succ"] = True
        data[t_name]["cnt"] += 1
        data[t_name]["pen"] += srv["fail"] * 20
        mins = (q_sec - start_sec) // 60
        data[t_name]["pen"] += mins

result = []
for t_name, d in data.items():
    result.append((d["cnt"], d["pen"], t_name))

result.sort(key=lambda x: (-x[0], x[1], x[2]))
rank = 1
for i, (cnt, pen, t_name) in enumerate(result):
    if i > 0 and result[i - 1][0] == cnt and result[i - 1][1] == pen:
        print(f"{rank} \"{t_name}\" {cnt} {pen}")
    else:
        rank = i + 1
        print(f"{rank} \"{t_name}\" {cnt} {pen}")