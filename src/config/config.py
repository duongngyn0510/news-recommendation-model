class BaseConfig():
    batch_size = 32
    num_words_title = 15
    num_words_body = 40


class NAMLConFig(BaseConfig):
    word_embedding_dim = 300
    num_filters = 300
    window_size = 3
    query_vector_dim = 200
    drop_prob = 0.1
    num_words = 1 + 70975
    num_categories = 1 + 274
    category_embedding_dim = 100
    num_click_news = 20
