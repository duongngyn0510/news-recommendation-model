import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(
            self,
            query_vector_dim,
            context_vector_dim
    ):
        """
        Additive attention for NAML encoder
        """
        super(AdditiveAttention, self).__init__()
        self.query_vector = nn.Parameter(torch.empty(query_vector_dim).uniform_(-0.1, 0.1))
        self.linear = nn.Linear(context_vector_dim, query_vector_dim)

    def forward(self, candidate_vector):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim (b, n, d)
        Returns:
            (shape): batch_size, candidate_vector_dim (b, d)
        """
        # batch_size, candidate_size, query_vector_dim (b, n, d)
        temp = torch.tanh(self.linear(candidate_vector))

        # batch_size, candidate_size (b, n)
        unnormalized_weights = torch.einsum('b n d, d -> b n', temp, self.query_vector)
        
        # batch_size, candidate_size (b, n)
        weights = F.softmax(unnormalized_weights, dim=1)

        # batch_size, candidate_vector_dim (b, d)
        output = torch.einsum('b n, b n d -> b d', weights, candidate_vector)
        return output




