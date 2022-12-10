# config.py

class Config(object):
    N = 6 #6 in Transformer Paper
    d_model = 256 #512 in Transformer Paper
    d_ff = 512 #2048 in Transformer Paper
    h = 8
    dropout = 0.2
    output_size = 295
    lr = 0.0005
    max_epochs = 60
    batch_size = 256
    max_sen_len = 60
    gamma = 0.5
    model_name = "jackaduma/SecBERT" #bert-base-uncased jackaduma/SecBERT
    num_channels = 100
    kernel_size = [3,4,5]
    lstm_hiddens = 768
    lstm_layers = 2