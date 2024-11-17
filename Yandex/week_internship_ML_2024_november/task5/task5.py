import hashlib

def estimate_unique_queries(n, queries):
    seen = [0] * 100000
    unique_count = 0

    for query in queries:
        query_hash = int(hashlib.sha256(query.encode('utf-8')).hexdigest(), 16) % 100000
        if seen[query_hash] == 0:
            seen[query_hash] = 1
            unique_count += 1

    return unique_count

n = int(input().strip())
queries = [input().strip() for _ in range(n)]
unique_count = estimate_unique_queries(n, queries)
print(unique_count)
