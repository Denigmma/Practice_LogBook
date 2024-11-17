with open('input.txt', 'r') as infile:
    R, C = map(int, infile.readline().strip().split())
    crossword = [infile.readline().strip() for _ in range(R)]

words = []

# check for line
for i in crossword:
    init_word=""
    for j in range(C):
        if i[j]=="#":
            if len(init_word)>=2:
                words.append(init_word)
            init_word = ""
        else:
            init_word+=i[j]
    if len(init_word) >= 2:
        words.append(init_word)

#check for col
for i in range(C):
    init_word = ""
    for j in range(R):
        if crossword[j][i] == "#":
            if len(init_word) >= 2:
                words.append(init_word)
            init_word = ""
        else:
            init_word += crossword[j][i]
    if len(init_word) >= 2:
        words.append(init_word)


result=min(words)
with open('output.txt', 'w') as outfile:
    outfile.write(result)