class BaseConfig():
    batch_size = 32
    num_words_title = 15
    num_words_body = 40
    word_embedding_dim = 200
    num_words = 1 + 70975
    num_click_news = 20
    num_categories = 1 + 274
    query_vector_dim = 200
    drop_prob = 0.1


class NAMLConFig(BaseConfig):
    num_filters = 300
    window_size = 3
    category_embedding_dim = 100


class NRMSConFig(BaseConfig):
    num_heads = 8


class LSTURConFig(BaseConfig):
    num_filters = 300
    window_size = 3
    category_embedding_dim = 100
    long_short_term_method = 'init'
    num_recent_clicked_user = 25