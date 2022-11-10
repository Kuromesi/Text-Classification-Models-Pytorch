import torch
import spacy
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
from torchtext.data.functional import to_map_style_dataset


class MyDataset(Dataset):
    def __init__(self, label_data): 
        self.label, self.data = label_data
        self.length = len(self.label)
    
    def __getitem__(self, index):
        return self.label[index], self.data[index]
    
    def __len__(self): 
        return self.length

def parse_label(label):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form '__label__2'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        return int(label.strip()[-1])

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(_label)
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    return label_list.to(device), text_list.to(device), offsets.to(device)

NLP = spacy.load('en_core_web_trf')
vocab = NLP.vocab

# Some pipelines
tokenizer = lambda sent: [x.lemma_.lower() for x in NLP(sent) if x.lemma_.lower() != " "]


# Some definitions
train_file = './data/ag_news.train'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(train_file, 'r') as datafile:     
            data = [line.strip().split(',', maxsplit=1) for line in datafile]
            data_text = list(map(lambda x: x[1], data))
            data_label = list(map(lambda x: parse_label(x[0]), data))
train = list(zip(data_label, data_text))
train_iter = to_map_style_dataset(iter(train))
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)
for idx, (label, text, offsets) in enumerate(dataloader):
    print(1)
# test = "Sad asd asd asasd"
# vocab = NLP.vocab
# l = list(vocab.strings)
# tmp = vocab['happy']
# print(l.index('happy'))
# print(tokenizer(test))