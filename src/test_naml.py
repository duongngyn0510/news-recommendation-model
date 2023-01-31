from NAML.news_encoder import NewsEncoder
from NAML.users_encoder import UserEncoder
from config.config import NAMLConFig
import torch
import random


config = NAMLConFig()
batch_size = config.batch_size
num_words_title = config.num_words_title
num_words_body = config.num_words_body
num_click_news = config.num_click_news
num_filters = config.num_filters

news_encoder = NewsEncoder(config)
user_encoder = UserEncoder(config)

if __name__ == '__main__':
    # create dummy news 
    news = {
        "title": torch.empty((batch_size, num_words_title), dtype=torch.int32).random_(config.num_words),
        "body": torch.empty((batch_size, num_words_body), dtype=torch.int32).random_(config.num_words),
        "category": torch.empty(batch_size, dtype=torch.int32).random_(config.num_categories),
        "sub_category": torch.empty(batch_size, dtype=torch.int32).random_(config.num_categories),
    }

    news_vectors = news_encoder(news)
    assert news_vectors.shape == (batch_size, num_filters)

    # create dummy news clicked by users
    clicked_news_vector = torch.zeros(batch_size, num_click_news, num_filters)
    for user in range(batch_size):
        for new in range(num_click_news):
            i = random.randint(0, batch_size - 1)
            random_news_vector = news_vectors[i]
            clicked_news_vector[user, new, :] = random_news_vector

    user_vector = user_encoder(clicked_news_vector)
    assert user_vector.shape == (batch_size, num_filters)

    print("Compatible shape!")