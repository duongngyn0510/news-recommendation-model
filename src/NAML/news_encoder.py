import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from attention.additive import AdditiveAttention


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TextEncoder(nn.Module):
    def __init__(
            self, 
            word_embedding, 
            word_embedding_dim, 
            num_filters, 
            window_size,
            query_vector_dim,
            drop_prob
        ):
        super(TextEncoder, self).__init__()
        self.word_embedding = word_embedding
        self.drop_prob = drop_prob
        self.cnn = nn.Conv2d(
            1, 
            num_filters, 
            (window_size, word_embedding_dim), 
            padding=(int((window_size - 1) / 2), 0)
        )
        self.additive = AdditiveAttention(query_vector_dim, num_filters)

    def forward(self, text):
        # batch_size, num_words, embedding_dim (b, n, e)
        text_vector = F.dropout(
            self.word_embedding(text),
            p=self.drop_prob)
        
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
        final_text_vector = self.additive(activated_text_vector)

        return final_text_vector


class ElementEncoder(nn.Module):
    def __init__(self, word_embedding, input_dim, output_dim):
        super(ElementEncoder, self).__init__()
        self.embedding = word_embedding
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, element):
        return F.relu(self.linear(self.embedding(element)))
    

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
        text_encoders_candidates = ['title', 'body']
        self.text_encoders = nn.ModuleDict({
            name: TextEncoder(
                self.word_embedding,
                config.word_embedding_dim,
                config.num_filters,
                config.window_size,
                config.query_vector_dim,
                config.drop_prob
            )
            for name in text_encoders_candidates
        })

        self.category_embedding = nn.Embedding(
            config.num_categories, 
            config.category_embedding_dim
            )
        elements_encoders_candidates = ['category', 'sub_category']
        self.elements_encoders = nn.ModuleDict({
            name: ElementEncoder(
                self.category_embedding,
                config.category_embedding_dim,
                config.num_filters
            )
            for name in elements_encoders_candidates 
        })
        
        self.final_attention = AdditiveAttention(config.query_vector_dim, config.num_filters)

    def forward(self, news):
        """
        Args:
            news:
            {
                "title": batch_size * num_words_title
                "body": batch_size * num_words_body
                "category": batch_size,
                "sub_category": batch_size
            }
        Returns
            (shape): batch_size, 
        """
        text_vectors = [
            encoder(news[name].to(device)) for name, encoder in self.text_encoders.items()
        ]
        element_vectors = [
            encoder(news[name].to(device)) for name, encoder in self.elements_encoders.items()
        ]
        vectors = text_vectors + element_vectors
        assert len(vectors) == len(news)
        # batch_size, 4, num_filters
        vectors = torch.stack(vectors, dim=1)
        final_news_vector = self.final_attention(vectors)
        return final_news_vector