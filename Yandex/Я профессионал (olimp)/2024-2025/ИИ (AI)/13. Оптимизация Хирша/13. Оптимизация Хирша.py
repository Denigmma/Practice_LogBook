import json
import re
from collections import defaultdict

# Load the data from submit.json
with open('submit.json', 'r') as f:
    papers = json.load(f)


# Function to tokenize a title
def tokenize(title):
    # Remove punctuation and split by spaces
    title = re.sub(r'[^\w\s]', '', title).lower()
    return title.split()


# Calculate suspiciousness of a group of papers
def calculate_suspiciousness(group_tokens):
    all_tokens = set()
    token_count = defaultdict(int)

    for tokens in group_tokens:
        all_tokens.update(tokens)
        for token in tokens:
            token_count[token] += 1

    suspiciousness = 0
    for token in all_tokens:
        if token_count[token] < len(group_tokens):
            suspiciousness += 1
    return suspiciousness


# Sort papers by citation count in descending order
papers.sort(key=lambda x: x['citations'], reverse=True)

# Grouping strategy with a maximum number of groups
max_groups = 5  # Set a reasonable limit for the number of groups
groups = []
group_tokens = []
group_citations = []

for paper in papers:
    paper_tokens = tokenize(paper['title'])
    if len(groups) < max_groups:
        # Try to add the paper to an existing group with the lowest suspiciousness
        min_suspiciousness = float('inf')
        min_group_index = -1
        for i, tokens in enumerate(group_tokens):
            current_suspiciousness = calculate_suspiciousness(tokens + [paper_tokens])
            if current_suspiciousness <= 42 and current_suspiciousness < min_suspiciousness:
                min_suspiciousness = current_suspiciousness
                min_group_index = i
        if min_group_index != -1:
            groups[min_group_index].append(paper)
            group_tokens[min_group_index].append(paper_tokens)
            group_citations[min_group_index] += paper['citations']
        else:
            # Start a new group
            groups.append([paper])
            group_tokens.append([paper_tokens])
            group_citations.append(paper['citations'])
    else:
        # If the maximum number of groups is reached, try to find the best group to add the paper
        min_suspiciousness = float('inf')
        min_group_index = -1
        for i, tokens in enumerate(group_tokens):
            current_suspiciousness = calculate_suspiciousness(tokens + [paper_tokens])
            if current_suspiciousness <= 42 and current_suspiciousness < min_suspiciousness:
                min_suspiciousness = current_suspiciousness
                min_group_index = i
        if min_group_index != -1:
            groups[min_group_index].append(paper)
            group_tokens[min_group_index].append(paper_tokens)
            group_citations[min_group_index] += paper['citations']
        else:
            # If no suitable group is found, start a new group (though this should not happen with max_groups)
            groups.append([paper])
            group_tokens.append([paper_tokens])
            group_citations.append(paper['citations'])

# Optimize groups by trying to merge groups with low suspiciousness
merged = True
while merged:
    merged = False
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            combined_tokens = group_tokens[i] + group_tokens[j]
            if calculate_suspiciousness(combined_tokens) <= 42:
                # Merge groups i and j
                groups[i].extend(groups[j])
                group_tokens[i].extend(group_tokens[j])
                group_citations[i] += group_citations[j]
                del groups[j]
                del group_tokens[j]
                del group_citations[j]
                merged = True
                break
        if merged:
            break

# Ensure the number of groups does not exceed max_groups
if len(groups) > max_groups:
    # Remove groups with the lowest total citations until the number of groups is max_groups
    while len(groups) > max_groups:
        min_citations_index = group_citations.index(min(group_citations))
        del groups[min_citations_index]
        del group_tokens[min_citations_index]
        del group_citations[min_citations_index]

# Reassign group indices to papers
for group_index, group in enumerate(groups):
    for paper in group:
        paper['group'] = group_index

with open('submit_up.json', 'w') as f:
    json.dump(papers, f, indent=4)

# Calculate the Hirsch index
citation_counts = sorted(group_citations, reverse=True)
h_index = 0
for i, citations in enumerate(citation_counts):
    if citations >= i + 1:
        h_index = i + 1
    else:
        break

print(f"Calculated Hirsch Index: {h_index}")

for group_index, tokens in enumerate(group_tokens):
    suspiciousness = calculate_suspiciousness(tokens)
    print(f"Group {group_index}: Suspiciousness = {suspiciousness}, Citations = {group_citations[group_index]}")