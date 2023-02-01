import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch


class UserEncoder(nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        self.gru = nn.GRU(
            config.num_filters * 3,
            config.num_filters * 3 if config.long_short_term_method == 'init'
            else config.num_filters
        )

    def forward(self, user, clicked_news_list):
        """
        Args:
            user:
                init: batch_size, num_filters * 3
                con: batch_size, num_filters 
            clicked_news_list: (len) batch_size 
        Returns:
            user_vector: (shape) batch_size, num_filters * 3
        """
        padded_clicked_user_vector = rnn_utils.pad_sequence(
                clicked_news_list, 
                batch_first=True
            )
        clicked_news_length = torch.tensor([len(nums_news) for nums_news in clicked_news_list])
        clicked_news_length[clicked_news_length == 0] = 1
        packed_clicked_user_vector = rnn_utils.pack_padded_sequence(
            padded_clicked_user_vector, 
            clicked_news_length,
            batch_first=True,
            enforce_sorted=False
        )
        if self.config.long_short_term_method == 'init':
            # batch_size, num_filters * 3
            _, last_hidden_user_state = self.gru(
                packed_clicked_user_vector,
                user.unsqueeze(dim=0)
            )
            return last_hidden_user_state.squeeze(dim=0)
        else:
            _, short_term_user_vector = self.gru(
                packed_clicked_user_vector
            )
            return torch.concat((user, short_term_user_vector.squeeze(dim=0)), dim=1)



