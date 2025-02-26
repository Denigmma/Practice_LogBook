import math

def determine_action(t1, x1, y1, t2, x2, y2):
    time_diff = t2 - t1
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    if time_diff < 100:
        return 1
    elif 100 <= time_diff <= 1000 and distance <= 50:
        return 2
    elif time_diff > 1000 and distance <= 50:
        return 3
    elif time_diff >= 100 and distance > 50:
        return 4

def main():
    num_tests = int(input())

    results = []

    for _ in range(num_tests):
        start_time, start_x, start_y = map(int, input().split())
        end_time, end_x, end_y = map(int, input().split())

        action = determine_action(start_time, start_x, start_y, end_time, end_x, end_y)
        results.append(action)

    for result in results:
        print(result)

main()
