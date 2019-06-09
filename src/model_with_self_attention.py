import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

# Reference: https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm
# ↑ difference between output and hidden!!!

class Self_Attention(nn.Module):
    def __init__(self, query_dim):
        # assume: query_dim = key/value_dim
        super(Self_Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, key, value):
        # query == hidden: (batch_size, hidden_dim * 2)
        # key/value == gru_output: (sentence_length, batch_size, hidden_dim * 2)
        query = query.unsqueeze(1) # (batch_size, 1, hidden_dim * 2)
        key = key.transpose(0, 1).transpose(1, 2) # (batch_size, hidden_dim * 2, sentence_length)

        # bmm: batch matrix-matrix multiplication
        attention_weight = torch.bmm(query, key) # (batch_size, 1, sentence_length)
        attention_weight = F.softmax(attention_weight.mul_(self.scale), dim=2) # normalize sentence_length's dimension

        value = value.transpose(0, 1) # (batch_size, sentence_length, hidden_dim * 2)
        attention_output = torch.bmm(attention_weight, value) # (batch_size, 1, hidden_dim * 2)
        attention_output = attention_output.squeeze(1) # (batch_size, hidden_dim * 2)

        return attention_output, attention_weight.squeeze(1)


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, d_rate):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, 
                        bidirectional=True, dropout=d_rate)
        self.dense = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = nn.Dropout(d_rate)
        self.attention = Self_Attention(2 * hidden_dim)
    
    def forward(self, x):
        # x: (sentence_length, batch_size)

        embedded = self.dropout(self.embedding(x))
        # embedded: (sentence_length, batch_size, embedding_dim)

        gru_output, hidden = self.gru(embedded)
        # gru_output: (sentence_length, batch_size, hidden_dim * 2)
        ## depth_wise
        # hidden: (num_layers * 2, batch_size, hidden_dim)
        ## ordered: [f_layer_0, b_layer_0, ...f_layer_n, b_layer n]

        # concat the final output of forward direction and backward direction
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        # hidden: (batch_size, hidden_dim * 2)

        rescaled_hidden, attention_weight = self.attention(query=hidden, key=gru_output, value=gru_output)
        output = self.dense(rescaled_hidden)

        return output.squeeze(1), attention_weight