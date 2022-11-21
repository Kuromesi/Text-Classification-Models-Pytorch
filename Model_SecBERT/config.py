# config.py

class Config(object):
    N = 6 #6 in Transformer Paper
    d_model = 52000 #512 in Transformer Paper
    d_ff = 512 #2048 in Transformer Paper
    h = 8
    dropout = 0.1
    output_size = 11
    lr = 0.0005
    max_epochs = 40
    batch_size = 32
    max_sen_len = 60
    gamma = 0.5
    model_name = "bert-base-uncased" #bert-base-uncased jackaduma/SecBERT
    num_channels = 100
    kernel_size = [3,4,5]
    dropout_keep = 0.8