import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, d_rate):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, 
                        bidirectional=True, dropout=d_rate)
        self.dense = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = nn.Dropout(d_rate)
    
    def forward(self, x):
        # x: (sentence_length, batch_size)

        embedded = self.dropout(self.embedding(x))
        # embedded: (sentence_length, batch_size, embedding_dim)

        gru_output, hidden = self.gru(embedded)
        # gru_output: (sentence_length, batch_size, hidden_dim * 2)
        # hidden: (num_layers * 2, batch_size, hidden_dim)
        ## ordered: [f_layer_0, b_layer_0, ...f_layer_n, b_layer n]
        
        # concat the final output of forward direction and backward direction
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        # hidden: (batch_size, hidden_dim * 2)

        output = self.dense(hidden)

        return output