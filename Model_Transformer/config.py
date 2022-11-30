# config.py

class Config(object):
    N = 6 #6 in Transformer Paper
    d_model = 256 #512 in Transformer Paper
    d_ff = 512 #2048 in Transformer Paper
    h = 8
    dropout = 0.2
    output_size = 14
    lr = 0.0002
    max_epochs = 60
    batch_size = 16
    max_sen_len = 200
    gamma = 0.5
    model_name = "bert-base-uncased" #bert-base-uncased jackaduma/SecBERT
    num_channels = 100
    kernel_size = [3,4,5]
    lstm_hiddens = 768
    lstm_layers = 2