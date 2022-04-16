import torch
from torch import nn
from gensim.models import Word2Vec

class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path): # First define the property of some parameters
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        # Load the saved model
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size # vector length

    def add_embedding(self, word):
        '''
        Add word into embedding, and give a representation vector that produced randomly
        '''
        vector = torch.empty(1, self.embedding_dim) 
        torch.nn.init.uniform_(vector) # Produced randomly
        self.word2idx[word] = len(self.word2idx) # put corresponding index into wordidx
        self.idx2word.append(word) # put word into idx2word
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0) # add new vector into embedding matrix
   
    def make_embedding(self, load=True):
        print("Get embedding ...")

        if load:
            print("loading word2vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError

        # word2idx is a dictionary
        # idx2word is a list
        # word2vector is a list
        for i, word in enumerate(self.embedding.wv.vocab):
            # print('get words #{}'.format(i+1), end='/r')
            #e.g. self.word2index['aaa'] = 1 
            #e.g. self.index2word[1] = 'aaa'
            #e.g. self.vectors[1] = 'aaa' vector
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # Add "" and "" into embeddings
        self.add_embedding("")
        self.add_embedding("")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # Get uniform length
        if len(sentence) > self.sen_len:          # Truncate
            sentence = sentence[:self.sen_len]
        else:                                     # Add ""
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx[""])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        # Transform the word in sentence into corresponding index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            # print('sentence count #{}'.format(i+1), end='/r')
            sentence_idx = []
            for word in sen:
                if word in self.word2idx.keys():
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx[""])
            # Get uniform length
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        # Transform label to tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)


# Function to get token ids for a list of texts 
def bert_tokenize_fn(text_list, tokenizer, seq_len):   
    all_input_ids = []    
    for text in text_list:
        input_ids = tokenizer.encode(
                        text,                      
                        add_special_tokens = True,   # Add special tokens，including CLS and SEP
                        max_length = seq_len,        # Limit the max sequence length
                        padding = 'max_length',      # Padding to max length  
                        return_tensors = 'pt',       # return type is pytorch tensor
                        truncation = True
                    )
        all_input_ids.append(input_ids)    
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids