from attention.additive import AdditiveAttention
import torch.nn as nn


class UserEncoder(nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.additive = AdditiveAttention(
            config.query_vector_dim,
            config.num_filters
        )

    def forward(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_click_news, num_filters
        Returns:
            user vector : batch_size, num_filters
        """
        user_vector = self.additive(clicked_news_vector)
        return user_vector