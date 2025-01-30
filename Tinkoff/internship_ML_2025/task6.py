from itertools import combinations

def is_non_degenerate(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) != (p3[0] - p1[0]) * (p2[1] - p1[1])

def max_happy_triplets(n, houses):
    valid_triplets = []
    used = set()

    for triplet in combinations(range(n), 3):
        i, j, k = triplet
        if is_non_degenerate(houses[i], houses[j], houses[k]):
            if i not in used and j not in used and k not in used:
                valid_triplets.append(triplet)
                used.update(triplet)

    return len(valid_triplets)

n = int(input())
houses = [tuple(map(int, input().split())) for _ in range(n)]
print(max_happy_triplets(n, houses))
