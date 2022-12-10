import torch
from utils import *
from model import *
from config import Config

def loadLabels(path):
    with open(path, 'r') as f:
        labels = []
        for line in f:
            labels.append(line.strip())
    return labels

config = Config()
dataset = Dataset(config)
train_file = './data/capec/train.txt'
test_file = './data/capec/test.txt'

dataset = Dataset(config)
model = load_model('ckpts/transformer-cve.pkl')
text = "Adobe Photoshop version 22.1 (and earlier) is affected by a heap buffer overflow vulnerability when handling a specially crafted font file. Successful exploitation could lead to arbitrary code execution. Exploitation of this issue requires user interaction in that a victim must open a malicious file."
text_vec = dataset.text2vec(text)
pred = model(text_vec['input_ids'].cuda())
pred = pred.cpu().data
pred = torch.where(pred >= 0.3, 1, pred)
pred = torch.where(pred < 0.3, 0, pred)
pred = pred.tolist()[0]
labels = loadLabels("data/CVE2Technique/classification.labels")
for i in range(len(labels)):
    if pred[i] == 1.0:
        print(labels[i])
# print(pred)