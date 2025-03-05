import csv

def classify_word(word):
    word = word.strip()
    return "A" if len(word) % 2 == 1 else "B"

with open("data/test.csv", "r", encoding="utf-8", newline="") as infile:
    reader = csv.DictReader(infile)
    words = [row["word_name"] for row in reader]

with open("result.csv", "w", encoding="utf-8", newline="") as outfile:
    # Используем lineterminator="\n", чтобы строки заканчивались ровно одним переводом строки
    writer = csv.writer(outfile, lineterminator="\n")
    writer.writerow(["word_name", "class_name"])
    for word in words:
        writer.writerow([word, classify_word(word)])
    outfile.write("\n")
