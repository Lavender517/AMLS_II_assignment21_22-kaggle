from sympy import arg
import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from code.data_load import load_data, TwitterDataset
from code.data_preprocess import Preprocess, bert_tokenize_fn
from code.models import LSTM_Net, TextCNN
from code.train_test import training, testing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sen_len', type=int, default=30, help='The specified sentence length')
parser.add_argument('--fixed_embedding', type=bool, default=True, help='If needed to fix embedding during training process')
parser.add_argument('--embedding_dim', type=int, default=200, help='Embedded feature dimenstion')
parser.add_argument('--batch_size', type=int, default=64, help='The data to be included in each epoch')
parser.add_argument('--num_workers', type=int, default=16, help='How many subprocesses to use for data loading')
parser.add_argument('--n_epochs', type=int, default=20, help='Training epochs = samples_num / batch_size')
parser.add_argument('--lr', type=float, default=5e-6, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=5e-2, help='Regularization coefficient, usually use 5 times, for example: 1e-4/5e-4/1e-5/5e-5')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout coefficient')
parser.add_argument('--device', type=int, default=5, help='The specified GPU number to be used')
parser.add_argument('--early_stop_TH', type=int, default=6, help='The threshold value of the valid_loss continue_bigger_num in early stopping criterion')
parser.add_argument('--model', type=str, default='BERT', help='The specifc deep learning model to be chosed')
args = parser.parse_args()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device('cuda'):
    device = args.device
code_path = './code'
w2v_path = os.path.join(code_path, 'w2v_all_' + str(args.embedding_dim) + '.model')


def train():
    print("Loading training data ...")
    train_x, train_y, _ = load_data()

    if args.model == "BERT":
        # Preprocessing on Testing Dataset
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        train_x = bert_tokenize_fn(train_x, tokenizer, args.sen_len) # input_ids
        train_y = torch.LongTensor(train_y)
        # train_y = torch.nn.functional.one_hot(train_y)

        # Build Training Model
        train_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1, output_attentions=False, output_hidden_states=False)
        print("Using BERT Model")

        optimizer = optim.AdamW(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Preprocessing on Training Dataset
        train_pp = Preprocess(train_x, args.sen_len, w2v_path=w2v_path)
        embedding = train_pp.make_embedding(load=True)
        train_x = train_pp.sentence_word2idx()
        train_y = train_pp.labels_to_tensor(train_y)

        # Build Training Model
        if args.model == "LSTM":
            train_model = LSTM_Net(embedding, embedding_dim=args.embedding_dim, hidden_dim=200, num_layers=2, dropout=args.dropout, fix_embedding=args.fixed_embedding)
            print("Using Bi-LSTM Model")

        elif args.model == "CNN":
            train_model = TextCNN(embedding, kernel_sizes=[3, 4, 5], dropout=args.dropout, num_channels=[200, 200, 200])
            print("Using TextCNN Model")
        
        else:
            print("Model type choice ERROR! Please choose one from ['LSTM', 'CNN', 'BERT'].")

        optimizer = optim.Adam(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_model = train_model.to(device)
    # Split Dataset into train dataset and validation dataset
    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, random_state=42, test_size=0.2)
    # Construct DataSet and DataLoader
    train_dataset = TwitterDataset(X = X_train, y = y_train)
    val_dataset = TwitterDataset(X = X_val, y = y_val)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

    criterion = nn.BCELoss() # Use binary cross entropy loss
    # criterion = nn.CrossEntropyLoss() # Use binary cross entropy loss
    total_steps = len(train_loader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    training(args.model, args.batch_size, args.n_epochs, criterion, optimizer, scheduler, train_loader, val_loader, train_model, device, args.early_stop_TH)
    print("Training process finished!")

def test():
    print("Loading testing data ...")
    _, _, test_x = load_data()
    
    # Preprocessing on Testing Dataset
    if args.model == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        test_x = bert_tokenize_fn(test_x, tokenizer, args.sen_len) # input_ids
    else:
        test_pp = Preprocess(test_x, args.sen_len, w2v_path=w2v_path)
        test_x = test_pp.sentence_word2idx()

    test_dataset = TwitterDataset(X = test_x, y = None)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

    print('Load model ...')
    saved_model = torch.load('./models/ckpt_81.38020833333334.model')
    outputs = testing(args.model, test_loader, saved_model, device)

    # Save to csv
    test_org = pd.read_csv('./Datasets/test.csv')
    result = pd.DataFrame({"text": test_org['text'].values, "label": outputs})
    print("Save prediction results ...")
    result.to_csv('./Datasets/prediction.csv', index=False)
    print("Finish Predicting")

if __name__ == '__main__':
    print("Using hyperparameters as:", args)
    train()
    # test()