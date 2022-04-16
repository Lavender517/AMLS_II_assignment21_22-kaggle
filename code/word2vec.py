import os
import numpy as np
import pandas as pd
from gensim.models import word2vec
from data_load import load_data

def train_word2vec(x, vector_size):
    '''
    Train the model Word2Vec to get word embeddings
    x: input texts after tokenization
    '''
    # sg=1 choose skip-gram algorithm
    # vecotr_size is the layer numbers of neural network, min_count is to ignore the words with least frequency, worker is the threads number, 
    model = word2vec.Word2Vec(x, size=vector_size, window=10, min_count=5, workers=12, iter=10, sg=1)
    return model

if __name__ == '__main__':
    print("Loading training data and testing data ...")
    train_x, train_y, test_x = load_data()

    vector_size = 200
    model = train_word2vec(train_x + test_x, vector_size)
    
    print("Saving model with vector size =", vector_size)
    model.save(os.path.join('../models', 'w2v_all_' + str(vector_size) + '.model')) # Save pre-trained models