### answer=0.4808303

import random

def simulate_meeting(max_time=100, simulations=10000000):
    successful_meetings = 0

    for _ in range(simulations):
        position1 = 0
        position2 = 10
        for _ in range(max_time):
            position1 += random.choice([-1, 1])
            position2 += random.choice([-1, 1])
            if position1 == position2:
                successful_meetings += 1
                break
    return successful_meetings / simulations

probability = simulate_meeting()
print(probability)
