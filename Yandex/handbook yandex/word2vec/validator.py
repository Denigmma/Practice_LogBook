import torch
import torch.nn.functional as f
import numpy as np

checkpoint = torch.load("word2vec_manual.pth", map_location="cpu", weights_only=True)

W1 = checkpoint["W1"]           # [vocab_size, embedding_dim]
W2 = checkpoint["W2"]           # [embedding_dim, vocab_size]
word2idx = checkpoint["word2idx"]
idx2word = checkpoint["idx2word"]
embedding_dim = checkpoint["embedding_dim"]
print("Модель загружена")

def get_embedding(word):
    if word not in word2idx:
        print(f" !!Слова '{word}' нет в словаре!!")
        return None
    idx = word2idx[word]
    return W1[idx].unsqueeze(0)  # shape [1, embedding_dim]

def find_similar(word, top_k=5):
    word_vec = get_embedding(word)
    if word_vec is None:
        return

    all_embeddings = W1  # shape [V, embedding_dim]

    # косинусное сходство
    similarities = f.cosine_similarity(word_vec, all_embeddings, dim=1)
    sim_scores = similarities.numpy()
    top_indices = np.argsort(-sim_scores)[1: top_k + 1]

    print(f"\nБлижайшие слова к '{word}':")
    for idx in top_indices:
        print(f"{idx2word[idx]:10s} -> {sim_scores[idx]:.3f}")



if __name__ == "__main__":
    find_similar("муж", top_k=5)
    find_similar("жена", top_k=5)
    find_similar("штирлиц", top_k=5)
    find_similar("трусы", top_k=5)

