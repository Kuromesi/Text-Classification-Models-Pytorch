# utils.py

import random
import torch
# from torchtext.legacy import data
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import pickle



# class Dataset(object):
#     def __init__(self, config):
#         self.config = config
#         self.train_iterator = None
#         self.test_iterator = None
#         self.val_iterator = None
#         self.vocab = []
#         self.word_embeddings = {}
    
#     def parse_label(self, label):
#         '''
#         Get the actual labels from label string
#         Input:
#             label (string) : labels of the form '__label__2'
#         Returns:
#             label (int) : integer value corresponding to label string
#         '''
#         return int(label.strip()[-1])

#     def get_pandas_df(self, filename):
#         '''
#         Load the data into Pandas.DataFrame object
#         This will be used to convert data to torchtext object
#         '''
#         with open(filename, 'r') as datafile:     
#             data = [line.strip().split(',', maxsplit=1) for line in datafile]
#             data_text = list(map(lambda x: x[1], data))
#             data_label = list(map(lambda x: self.parse_label(x[0]), data))

#         full_df = pd.DataFrame({"text":data_text, "label":data_label})
#         return full_df
    

#     # !!!GARBAGE!!!
#     # def _load_data(self, train_file, test_file=None):
#     #     NLP = spacy.load('en_core_web_trf')
#     #     RATIO = 0.8
#     #     vocab = set()
        
#     #     # Load train & validate data and generate vocab
#     #     with open(train_file, 'r') as datafile:     
#     #         data = [line.strip().split(',', maxsplit=1) for line in datafile]
#     #         data_text = list(map(lambda x: x[1], data))
#     #         data_label = list(map(lambda x: self.parse_label(x[0]), data))
        
#     #     tokenizer = lambda sent: [x.lemma_.lower() for x in NLP(sent) if x.lemma_.lower() != " "]
#     #     data = []
#     #     for text in data_text:
#     #         t = tokenizer(text)
#     #         data.append(t)
#     #         vocab.update(tuple(t))
#     #     vocab = list(vocab)
#     #     UNK = len(vocab)
#     #     temp = list(zip(data_label, data))
#     #     random.shuffle(temp)
#     #     idx = int(len(temp) * RATIO)
        
#     #     word2id = lambda word: vocab.index(word) / UNK
#     #     train = []
#     #     validate = []
#     #     for i in range(idx):
#     #         train.append(tuple([temp[i][0], list(map(word2id, temp[i][1]))]))
#     #     for i in range(idx, len(temp)):
#     #         validate.append(tuple([temp[i][0], list(map(word2id, temp[i][1]))]))
        
#     #     # Load test data 
#     #     with open(test_file, 'r') as datafile:     
#     #         data = [line.strip().split(',', maxsplit=1) for line in datafile]
#     #         data_text = list(map(lambda x: x[1], data))
#     #         data_label = list(map(lambda x: self.parse_label(x[0]), data))
#     #     data = []
#     #     for text in data_text:
#     #         t = tokenizer(text)
#     #         data.append(t)
#     #     temp = list(zip(data_label, data))
#     #     test = []
#     #     for i in range(len(temp)):
#     #         test.append(tuple([temp[i][0], list(map(word2id, temp[i][1]))]))
            
#     #     print ("Loaded {} training examples".format(len(train)))
#     #     print ("Loaded {} test examples".format(len(test)))
#     #     print ("Loaded {} validation examples".format(len(validate)))

        
    
#     def load_data(self, train_file, test_file=None, val_file=None):
#         '''
#         Loads the data from files
#         Sets up iterators for training, validation and test data
#         Also create vocabulary and word embeddings based on the data
        
#         Inputs:
#             train_file (String): path to training file
#             test_file (String): path to test file
#             val_file (String): path to validation file
#         '''

#         NLP = spacy.load('en_core_web_trf')
#         vocab = NLP.vocab
        
#         tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        
#         # Creating Field for data
#         TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
#         LABEL = data.Field(sequential=False, use_vocab=False)
        
#         datafields = [("text",TEXT),("label",LABEL)]
        
#         # Load data from pd.DataFrame into torchtext.data.Dataset
#         train_df = self.get_pandas_df(train_file)
#         train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
#         train_data = data.Dataset(train_examples, datafields)
        
#         test_df = self.get_pandas_df(test_file)
#         test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
#         test_data = data.Dataset(test_examples, datafields)
        
#         # If validation file exists, load it. Otherwise get validation data from training data
#         if val_file:
#             val_df = self.get_pandas_df(val_file)
#             val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
#             val_data = data.Dataset(val_examples, datafields)
#         else:
#             train_data, val_data = train_data.split(split_ratio=0.8)
        
#         TEXT.build_vocab(train_data)
#         self.vocab = TEXT.vocab
        
#         self.train_iterator = data.BucketIterator(
#             (train_data),
#             batch_size=self.config.batch_size,
#             sort_key=lambda x: len(x.text),
#             repeat=False,
#             shuffle=True)
        
#         self.val_iterator, self.test_iterator = data.BucketIterator.splits(
#             (val_data, test_data),
#             batch_size=self.config.batch_size,
#             sort_key=lambda x: len(x.text),
#             repeat=False,
#             shuffle=False)
        
#         print ("Loaded {} training examples".format(len(train_data)))
#         print ("Loaded {} test examples".format(len(test_data)))
#         print ("Loaded {} validation examples".format(len(val_data)))


        

def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch[1].cuda()
        else:
            x = batch[1]
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch[0].numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score

def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)