from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import os
import sys
import re
from torch.utils.data import Dataset

def filter_text(text):
    '''
    Implement pre-processing operation to original texts
    text: raw input text contents, type: List
    '''
    text = text.lower()

    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'http?:\/\/.*[\r\n]*', '', text)

    #Replace &amp, &lt, &gt with &,<,> respectively
    text = text.replace(r'&amp;?', r'and')
    text = text.replace(r'&lt;', r'<')
    text = text.replace(r'&gt;', r'>')  

    #remove mentions
    text = re.sub(r"(?:\@)\w+", '', text)

    #remove non ascii chars
    text = text.encode("ascii", errors="ignore").decode()

    #remove hashtag sign
    #text = re.sub(r"#","",text) 

    #remove some puncts (except . ! ?)
    text = re.sub(r'[:"#$%&\*+,-/:;<=>@\\^_`{|}~]+', '', text)
    text = re.sub(r'[!]+', '!', text)
    text = re.sub(r'[?]+', '?', text)
    text = re.sub(r'[.]+', '.', text)
    text = re.sub(r"'", "", text)
    text = re.sub(r"\(", "", text)
    text = re.sub(r"\)", "", text)

    # Tokenization
    text = text.strip('/n').split(' ')
    return text

def load_data():
    '''
    Load origin data, then return pre-processed data
    '''
    train_org = pd.read_csv('./Datasets/train.csv')
    test_org = pd.read_csv('./Datasets/test.csv')
    train_org['text'] = train_org['text'].apply(filter_text)
    train_org = train_org[train_org['text'] != '']
    train_x = list(train_org['text'].values)
    train_y = list(train_org['target'].values)

    test_org['text'] = test_org['text'].apply(filter_text)
    test_x = list(test_org['text'])
    return train_x, train_y, test_x

class TwitterDataset(Dataset):
    """
    Expected data shape like: (data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)
    
    __len__ will return the number of data
    """
    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        if self.label is None: 
            return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)