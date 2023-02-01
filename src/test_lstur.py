from LSTUR.news_encoder import NewsEncoder
from LSTUR.users_encoder import UserEncoder
from config.config import LSTURConFig
import torch
import random
import numpy as np


config = LSTURConFig()
batch_size = config.batch_size
num_words_title = config.num_words_title
num_click_news = config.num_click_news
num_filters = config.num_filters
num_recent_clicked_user = config.num_recent_clicked_user

news_encoder = NewsEncoder(config)
user_encoder = UserEncoder(config)

if __name__ == '__main__':
    # create dummy news 
    news = {
        "title": torch.empty((batch_size, num_words_title), dtype=torch.int32).random_(config.num_words),
        "category": torch.empty(batch_size, dtype=torch.int32).random_(config.num_categories),
        "sub_category": torch.empty(batch_size, dtype=torch.int32).random_(config.num_categories),
    }

    news_vectors = news_encoder(news)
    assert news_vectors.shape == (batch_size, num_filters * 3)

    # create dummy news clicked by users
    random_num_recent_clicked = torch.empty(batch_size).random_(num_recent_clicked_user)
    user_list = []
    for user in range(batch_size):
        num = int(random_num_recent_clicked[user].item())
        indices = torch.LongTensor(np.random.choice(batch_size, num))
        user_list.append(torch.index_select(news_vectors, 0, indices))
        
    if config.long_short_term_method == 'init':
        user_long_term_vector = torch.randn(batch_size, num_filters * 3)
        user_vector = user_encoder(user_long_term_vector, user_list)
        assert user_vector.shape == (batch_size, num_filters * 3)
    else:
        user_long_term_vector = torch.randn(batch_size, num_filters)
        user_vector = user_encoder(user_long_term_vector, user_list)
        assert user_vector.shape == (batch_size, num_filters * 2)
    print("Compatible shape!")