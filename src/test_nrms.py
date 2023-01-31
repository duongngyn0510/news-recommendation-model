from NRMS.news_encoder import NewsEncoder
from NRMS.users_encoder import UserEncoder
from config.config import NRMSConFig
import torch
import random


config = NRMSConFig()
batch_size = config.batch_size
num_words_title = config.num_words_title
num_words_body = config.num_words_body
num_click_news = config.num_click_news
embedding_dim = config.word_embedding_dim

news_encoder = NewsEncoder(config)
user_encoder = UserEncoder(config)

if __name__ == '__main__':
    # create dummy news 
    news = {
        "title": torch.empty((batch_size, num_words_title), dtype=torch.int32).random_(config.num_words),
    }

    news_vectors = news_encoder(news)
    assert news_vectors.shape == (batch_size, embedding_dim)

    # create dummy news clicked by users
    clicked_news_vector = torch.zeros(batch_size, num_click_news, embedding_dim)
    for user in range(batch_size):
        for new in range(num_click_news):
            i = random.randint(0, batch_size - 1)
            random_news_vector = news_vectors[i]
            clicked_news_vector[user, new, :] = random_news_vector

    user_vector = user_encoder(clicked_news_vector)
    assert user_vector.shape == (batch_size, embedding_dim)

    print("Compatible shape!")