from torch.nn import MultiheadAttention
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.multihead_self_attention = MultiheadAttention(
            embedding_dim,
            num_heads
        )
        assert embedding_dim % num_heads == 0,  "embed_dim must be divisible by num_heads" 

        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, candidate_vector):
        """
        Args:
            candidate_vector: shape (batch_size, num_words, embedding_dim)
        Returns:
            context_vector: shape (batch_size, num_words, embedding_dim)
        """
        q = self.W_Q(candidate_vector)
        k = self.W_K(candidate_vector)
        v = self.W_V(candidate_vector)
        context_vector = self.multihead_self_attention(q, k, v)
        return context_vector