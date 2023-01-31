import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from einops import rearrange
from attention.additive import AdditiveAttention
from attention.multihead_self_attention import MultiHeadSelfAttention


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NewsEncoder(nn.Module):
    def __init__(self, config, pretrained_word_embedding=None):
        super(NewsEncoder, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(
                config.num_words, 
                embedding_dim=config.word_embedding_dim,
                padding_idx=0
            )
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding,
                freeze=False,
                padding_idx=0
            )
        self.multihead_self_attention = MultiHeadSelfAttention(
            config.word_embedding_dim,
            config.num_heads
        )
        self.additive = AdditiveAttention(
            config.word_embedding_dim,
            config.query_vector_dim
        )
        
    def forward(self, news):
        """
        Args:
            news:
            {
                "title": shape (batch_size, num_words_title)
            }
        Returns:
            final_vector_news: shape (batch_size, word_embedding_dim)
        """
        # batch_size, num_words, embedding_dim
        news_vector = F.dropout(
            self.word_embedding(news['title'].to(device)),
            p=self.config.drop_prob
        )
        # batch_size, num_words, embedding_dim
        multihead_news_vector, _ = self.multihead_self_attention(news_vector)
        final_vector_news = self.additive(multihead_news_vector)
        return final_vector_news