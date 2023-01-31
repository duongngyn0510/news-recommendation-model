from NRMS.news_encoder import NewsEncoder
from attention.multihead_self_attention import MultiHeadSelfAttention
from attention.additive import AdditiveAttention
import torch.nn as nn


class UserEncoder(nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.multihead_self_attention = MultiHeadSelfAttention(
            config.word_embedding_dim,
            config.num_heads
        )
        self.additive = AdditiveAttention(
            config.word_embedding_dim,
            config.query_vector_dim
        )

    def forward(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_click_news, embedding_dim
        Returns:
            user_vector : batch_size, embedding_dim
        """
        multihead_user_vector, _ = self.multihead_self_attention(clicked_news_vector)
        user_vector = self.additive(multihead_user_vector)
        return user_vector