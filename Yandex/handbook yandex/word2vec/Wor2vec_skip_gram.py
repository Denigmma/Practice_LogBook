import torch
import torch.nn.functional as F
import re
import itertools
import kagglehub
import os
import pandas as pd
import numpy as np
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# датасет с отзывами
# path = kagglehub.dataset_download("laytsw/reviews")
# print("Path to dataset files:", path)
#
# csv_path = os.path.join(path, "reviews.csv")
# df = pd.read_csv(csv_path, sep="\t")
# print(df.head())
#
# texts = df["review"].astype(str).tolist()
# texts = texts[:100]

# датасет с шутками
path = kagglehub.dataset_download("vsevolodbogodist/data-jokes")
print("Path to dataset files:", path)
csv_path = os.path.join(path, "dataset.csv")
df = pd.read_csv(csv_path, sep=",", quotechar='"')
print(df.head())
texts = df["text"].astype(str).tolist()
texts = texts[:500]




def tokenize(s):
    s = re.sub(r"[^а-я ]+", "", s.lower())
    return s.split()

tokens_list = [tokenize(t) for t in texts]
tokens = list(itertools.chain.from_iterable(tokens_list))

vocab = list(set(tokens))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
V = len(vocab)
print("Размер словаря:", V)


def one_hot(idx, vocab_size):
    vec = torch.zeros(vocab_size)
    vec[idx] = 1.0
    return vec


def generate_skipgram_pairs(tokens, window_size=2):
    pairs = []
    for center_pos in range(len(tokens)):
        center_word = tokens[center_pos]
        context_indices = list(range(max(0, center_pos - window_size), min(len(tokens), center_pos + window_size + 1)))
        context_indices.remove(center_pos)
        for ctx_pos in context_indices:
            pairs.append((center_word, tokens[ctx_pos]))
    return pairs

pairs = generate_skipgram_pairs(tokens, window_size=2)
training_data = [(word2idx[c], word2idx[ctx]) for c, ctx in pairs]


embedding_dim = 10
# Матрица весов для эмбеддингов (V x embedding_dim)
W1 = torch.randn(V, embedding_dim, requires_grad=True)
# Матрица весов для выхода (embedding_dim x V)
W2 = torch.randn(embedding_dim, V, requires_grad=True)
optimizer = torch.optim.SGD([W1, W2], lr=0.01)



epochs=1
for epoch in range(epochs):
    total_loss = 0
    for center_idx, context_idx in training_data:
        # One-hot вектор для центрального слова
        x = one_hot(center_idx, V)  # размер V

        # Forward Pass
        h = torch.matmul(x, W1)       # embedding вектор: размер embedding_dim
        u = torch.matmul(h, W2)       # логиты по всем словам: размер V
        y_pred = F.softmax(u, dim=0)  # предсказание вероятностей

        # Loss
        target = one_hot(context_idx, V)
        loss = -torch.sum(target * torch.log(y_pred + 1e-9))  # кросс-энтропия
        total_loss += loss.item()

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Эпоха {epoch+1}, loss = {total_loss:.4f}")


torch.save({
    "W1": W1.detach(),
    "W2": W2.detach(),
    "word2idx": word2idx,
    "idx2word": idx2word,
    "embedding_dim": embedding_dim
}, "word2vec_manual.pth")
print("model saved")

print("\nЭмбеддинги слов:")
for word, idx in word2idx.items():
    vec = W1[idx].detach().numpy()
    print(f"{word:7s} -> {vec.round(3)}")
