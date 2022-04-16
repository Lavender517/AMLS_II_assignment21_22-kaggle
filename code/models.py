import torch
from torch import nn
import random
import torch

class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding) # The embedding parameters are called from saved word2vec model
       
        #If fix_embedding is False, it will be trained as well
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        # Using Bi-LSTM model
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)  
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_dim*2, 1),
                                        nn.Sigmoid())

        # Initialization
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
            if type(m) == nn.LSTM:
                for param in m._flat_weights_names:
                    if "weight" in param:
                        nn.init.xavier_uniform_(m._parameters[param])


    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None) # x.shape = [64, 40, 400] = [batch_size, seq_len, 2*hidden_size]
        # Get the last hidden state from LSTM
        x = x[:, -1, :] # x.shape = [64, 400]
        x = self.classifier(x) # x.shape = [64, 1]
        return x



class TextCNN(nn.Module):
    def __init__(self, embedding, kernel_sizes, dropout, num_channels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))

        self.constant_embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(sum(num_channels), 1),
                                        nn.Sigmoid())
        # self.dropout = nn.Dropout(dropout)
        # self.decoder = nn.Linear(sum(num_channels), 1)
        # self.sigmoid = nn.Sigmoid()
        # The maximum time aggregation layer has no parameters, so this instance can be shared
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # Construct a multi-dimension convolutional layer
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embedding.size(1), c, k))

        for m in self.modules():
            if type(m) in (nn.Linear, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, inputs):
        # Join the two embedding layers along the vector dimension,
        # The output shape of each embedding layer is [batch_size, seq_len, embedding_dim] 连结起来
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # According to the input format of the 1-dim convolution layer, the tensors are rearranged so that the channels are the second dimension
        embeddings = embeddings.permute(0, 2, 1) # exchange dimensions, get tensor's shape as [batch_size, channel, 1]
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        # outputs = self.sigmoid(self.decoder(self.dropout(encoding)))
        outputs = self.classifier(encoding)
        return outputs