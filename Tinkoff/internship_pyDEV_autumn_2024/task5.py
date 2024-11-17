from datetime import datetime, timedelta
from collections import defaultdict

def result_hakaton():
    start_time=input().strip()
    n=int(input().strip())
    teams=defaultdict(lambda: {"servers": defaultdict(int), "success": 0, "punishment": 0})
    for i in range(n):
        record=input().strip().split()
        team_name=record[0].strip('"')
        event_time=record[1]
        server=record[2]
        result=record[3]
        if result == "PONG":
            continue
        minutes_since_start = time_diff(start_time, event_time)
        if result in {"FORBIDEN", "DENIED"}:
            teams[team_name]["servers"][server]+=1
        elif result=="ACCESSED":
            if teams[team_name]["servers"][server]>=0:
                punishment = teams[team_name]["servers"][server]*20
                teams[team_name]["punishment"]+=punishment+minutes_since_start
                teams[team_name]["success"]+=1
            teams[team_name]["servers"][server]=-1
    results=[]
    for team_name, data in teams.items():
        results.append((team_name, data["success"], data["punishment"]))
    results.sort(key=lambda x:(-x[1],x[2],x[0]))
    rank=1
    for idx,(team_name,success, punishment) in enumerate(results):
        if idx>0 and (results[idx-1][1],results[idx-1][2])==(success,punishment):
            print(f'{rank} "{team_name}" {success} {punishment}')
        else:
            rank=idx+1
            print(f'{rank} "{team_name}" {success} {punishment}')

def time_diff(start_time, event_time):
    fmt='%H:%M:%S'
    start=datetime.strptime(start_time, fmt)
    event=datetime.strptime(event_time, fmt)
    if event<start:
        event+=timedelta(days=1)
    tdelta = event-start
    return tdelta.seconds//60

result_hakaton()