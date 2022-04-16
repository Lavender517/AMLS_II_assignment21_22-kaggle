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
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)  
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_dim*2, 1),
                                         nn.Sigmoid() )

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
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(sum(num_channels), 1)
        # 最大时间汇聚层没有参数，因此可以共享此实例
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # 创建多个一维卷积层
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embedding.size(1), c, k))

        for m in self.modules():
            if type(m) in (nn.Linear, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, inputs):
        # 沿着向量维度将两个嵌入层连结起来，
        # 每个嵌入层的输出形状都是（批量大小，词元数量，词元向量维度）连结起来
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # 根据一维卷积层的输入格式，重新排列张量，以便通道作为第2维
        embeddings = embeddings.permute(0, 2, 1)
        # 每个一维卷积层在最大时间汇聚层合并后，获得的张量形状是（批量大小，通道数，1）
        # 删除最后一个维度并沿通道维度连结
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.sigmoid(self.decoder(self.dropout(encoding)))
        return outputs