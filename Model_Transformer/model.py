# Model.py

import torch
import torch.nn as nn
from copy import deepcopy
from train_utils import Embeddings,PositionalEncoding
from attention import MultiHeadedAttention
from encoder import EncoderLayer, Encoder
from feed_forward import PositionwiseFeedForward
import numpy as np
from utils import *

class Transformer(nn.Module):
    def __init__(self, config, src_vocab):
        super(Transformer, self).__init__()
        self.config = config
        self.best = 0
        
        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff
        
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        
        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        self.src_embed = nn.Sequential(Embeddings(config.d_model, src_vocab), deepcopy(position)) #Embeddings followed by PE

        # Convolution Layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=config.num_channels, kernel_size=config.kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(config.max_sen_len - config.kernel_size[0]+1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=config.num_channels, kernel_size=config.kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(config.max_sen_len - config.kernel_size[1]+1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=config.num_channels, kernel_size=config.kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(config.max_sen_len - config.kernel_size[2]+1)
        )

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.num_channels*len(self.config.kernel_size),
            self.config.output_size
        )
        
        self.fc1 = nn.Linear(
            self.config.d_model,
            self.config.d_model
        )
        self.fc2 = nn.Linear(
            self.config.d_model,
            self.config.output_size
        )
        
        # Softmax non-linearity
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded_sents = self.src_embed(x) # shape = (batch_size, sen_len, d_model)
        encoded_sents = self.encoder(embedded_sents)
        
        # Convert input to (batch_size, d_model) for linear layer
        final_feature_map = encoded_sents[:,0,:]
        a = x.size()
        b = embedded_sents.size()
        
        encoded_sents = encoded_sents.permute(0, 2, 1)
        c = encoded_sents.size()
        conv_out1 = self.conv1(encoded_sents).squeeze(2)
        conv_out2 = self.conv2(encoded_sents).squeeze(2)
        conv_out3 = self.conv3(encoded_sents).squeeze(2)
        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_out = self.fc(all_out)
        # final_out = self.fc1(final_feature_map)
        # final_out = self.fc(final_out)
        return self.sigmoid(final_out)
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        # Exponential
        # self.attenuation = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.gamma)
        
        # Step
        self.attenuation = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.config.max_epochs / 3), gamma=self.config.gamma)
        
    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
    
    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2
                
    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
            
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch[1].cuda()
                # y = (batch[0] - 1).type(torch.cuda.LongTensor)
                y = batch[0].cuda()
            else:
                x = batch[1].type(torch.LongTensor)
                # y = (batch[0] - 1).type(torch.LongTensor)
                y = batch[0]
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            if i % 5 == 0:
                print("Iter: {}".format(i+1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []
                
                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model(self, val_iterator)
                if (i  > self.config.max_epochs / 3 and self.best < val_accuracy):
                    save_model(self, 'ckpts/transformer.pkl')
                    self.best = val_accuracy
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()

                
        return train_losses, val_accuracies