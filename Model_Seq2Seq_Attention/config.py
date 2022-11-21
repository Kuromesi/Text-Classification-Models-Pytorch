# config.py

class Config(object):
    embed_size = 300
    hidden_layers = 2
    hidden_size = 32
    bidirectional = True
    output_size = 11
    max_epochs = 100
    lr = 0.5
    batch_size = 256
    dropout_keep = 0.8
    max_sen_len = None # Sequence length for RNN