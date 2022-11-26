# train.py

from utils import *
from model import *
from config import Config
import sys
import torch.optim as optim
from torch import nn
import torch

MAX_LENGTH = 80

def predict(model, x):
    y_pred = model(x)
    predicted = torch.where(y_pred >= 0.3, 1, y_pred)
    predicted = torch.where(predicted < 0.3, 0, predicted)
    return predicted

def seq2vec(tokenizer, seq):
    return tokenizer(seq, padding=True, truncation=True, return_tensors="pt", max_length=MAX_LENGTH)['input_ids']
    

if __name__=='__main__':
    model = load_model('./ckpts/best_model/transformer+3layerfc.pkl')
    tokenizer = AutoTokenizer.from_pretrained('jackaduma/SecBERT')
    seq = ["The WebDAV extension in Microsoft Internet Information Services (IIS) 5.1 and 6.0 allows remote attackers to bypass URI-based protection mechanisms, and list folders or read, create, or modify files, via a %c0%af (Unicode / character) at an arbitrary position in the URI, as demonstrated by inserting %c0%af into a \"/protected/\" initial pathname component to bypass the password protection on the protected\\ folder, aka \"IIS 5.1 and 6.0 WebDAV Authentication Bypass Vulnerability,\" a different vulnerability than CVE-2009-1122."]
    temp = []
    for s in seq:
        temp.append(s + " " + "<PAD> " * MAX_LENGTH)
    x = seq2vec(tokenizer, temp)
    x = x.cuda()
    y_pred = predict(model, x)
    print(y_pred)