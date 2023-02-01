import torch.nn as nn
import torch.nn.functional as F
import torch
from attention.additive import AdditiveAttention
from einops import rearrange


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TitleEncoder(nn.Module):
    def __init__(
            self, 
            word_embedding, 
            word_embedding_dim, 
            num_filters, 
            window_size,
            query_vector_dim,
            drop_prob
        ):
        super(TitleEncoder, self).__init__()
        self.word_embedding = word_embedding
        self.drop_prob = drop_prob
        self.cnn = nn.Conv2d(
            1,
            num_filters,
            (window_size, word_embedding_dim),
            padding=(int((window_size - 1) / 2), 0)
        )
        self.additive = AdditiveAttention(
            query_vector_dim, num_filters
        )

    def forward(self, text):
        # batch_size, num_words, embedding_dim
        text_vector = F.dropout(
            self.word_embedding(text),
            p=self.drop_prob
        )
        
        # batch_size, 1, num_words, embedding_dim (b, 1, n, e): reshape to match input shape of CONV2D
        text_vector = rearrange(text_vector, 'b n e -> b 1 n e')

        # batch_size, num_filters, num_words, 1 (b, c, n, 1)
        convoluted_text_vector = self.cnn(text_vector)

        # batch_size, num_filters, num_words (b, c, n)
        convoluted_text_vector = rearrange(convoluted_text_vector, 'b c n 1 -> b c n')

        # batch_size, num_filters, num_words (b, c, n)
        activated_text_vector = F.dropout(
            F.relu(convoluted_text_vector), 
            p=self.drop_prob)
        
        # batch_size, num_words, num_filters: reshape to match input shape of additive attention
        activated_text_vector = rearrange(activated_text_vector, 'b c n -> b n c')

        # batch_size, num_filters
        title_vector = self.additive(activated_text_vector)

        return title_vector


class CategoryEncoder(nn.Module):
    def __init__(self, word_embedding):
        super(CategoryEncoder, self).__init__()
        self.embedding = word_embedding

    def forward(self, element):
        category_vector = self.embedding(element)
        return category_vector
    

class NewsEncoder(nn.Module):
    def __init__(self, config, pretrained_word_embedding=None):
        super(NewsEncoder, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(
                config.num_words, 
                config.word_embedding_dim,
                padding_idx=0
            )
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, 
                freeze=False,
                padding_idx=0
            )
        
        self.title_encoder = TitleEncoder(
            self.word_embedding,
                config.word_embedding_dim,
                config.num_filters,
                config.window_size,
                config.query_vector_dim,
                config.drop_prob
        )
        
        category_embedding = nn.Embedding(
                config.num_categories, 
                config.num_filters,
                padding_idx=0
        )

        elements_encoders_candidates = ['category', 'sub_category']
        self.category_encoder = nn.ModuleDict({
            name: CategoryEncoder(category_embedding)
            for name in elements_encoders_candidates
        })

    def forward(self, news):
        """
        Args:
            news:
            {
                "title": shape (batch_size, num_words_title)
                "category": shape (batch_size),
                "sub_category": shape (batch_size)
            }
        Returns
            news_vector (shape): (batch_size, num_filters * 3)
        """
        title_vector = [self.title_encoder(news['title'].to(device))]
        category_vector = [
            encoder(news[name].to(device)) for name, encoder in self.category_encoder.items()
        ]
        vectors = title_vector + category_vector
        news_vector = torch.concat(vectors, dim=-1)
        return news_vector
