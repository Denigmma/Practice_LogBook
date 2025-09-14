import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)
# print("Torch:", torch.__version__)

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, context_idxs):
        """
        context_idxs: LongTensor [batch_size, context_len]
        """
        context_vecs = self.in_embed(context_idxs)        # [batch, context_len, D]
        v_context = context_vecs.mean(dim=1)              # [batch, D]

        # 2. логиты = скалярные произведения со всеми "output embeddings"
        #   out_embed.weight имеет shape [V, D]
        logits = torch.matmul(v_context, self.out_embed.weight.t())  # [batch, V]

        # 3. softmax будет применён внутри CrossEntropyLoss
        return logits


checkpoint = torch.load("word2vec_CBOW.pth", map_location=device, weights_only=False)

loaded_model = CBOWModel(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"]
).to(device)

loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_model.eval()

word2idx = checkpoint["word2idx"]
idx2word = checkpoint["idx2word"]

print("Модель и словари успешно загружены!")


def get_embedding(word, model, word2idx):
    """Возвращает embedding слова, shape [1, embedding_dim]"""
    if word not in word2idx:
        print(f"Слова '{word}' нет в словаре!")
        return None
    idx = torch.tensor([word2idx[word]], device=device)
    return model.in_embed(idx)  # [1, D]

def find_similar(word, model, word2idx, idx2word, top_k=5):
    word_vec = get_embedding(word, model, word2idx)
    if word_vec is None:
        return

    all_embeddings = model.in_embed.weight      # [V, D]

    similarities = F.cosine_similarity(word_vec, all_embeddings, dim=1)  # [V]
    sim_scores = similarities.detach().cpu().numpy()

    top_indices = np.argsort(-sim_scores)[1:top_k+1]

    print(f"\nБлижайшие слова к '{word}':")
    for idx in top_indices:
        print(f"{idx2word[idx]:15s} -> {sim_scores[idx]:.3f}")

if __name__ == "__main__":
    find_similar("муж", loaded_model, word2idx, idx2word, top_k=5)
    find_similar("жена", loaded_model, word2idx, idx2word, top_k=5)
    find_similar("штирлиц", loaded_model, word2idx, idx2word, top_k=5)
    find_similar("полицейский", loaded_model, word2idx, idx2word, top_k=5)
    find_similar("мент", loaded_model, word2idx, idx2word, top_k=5)

