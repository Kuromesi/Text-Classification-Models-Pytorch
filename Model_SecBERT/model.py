# Model.py

import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from utils import *
from transformers import BertForSequenceClassification, BertConfig, BertModel, AutoModelForMaskedLM, BertPreTrainedModel
from transformers import RobertaConfig, RobertaModel
from encoder import EncoderLayer, Encoder
from train_utils import Embeddings,PositionalEncoding
from attention import MultiHeadedAttention
from feed_forward import PositionwiseFeedForward

class Classifier(BertPreTrainedModel):
    def __init__(self, config):
        
        bertConfig = RobertaConfig.from_pretrained(config.model_name, hidden_dropout_prob=config.dropout)
        super(Classifier, self).__init__(bertConfig)
        # config = BertConfig.from_pretrained(config.model_name, hidden_dropout_prob=config.dropout)
        self.src_embed = RobertaModel.from_pretrained(config.model_name, config=bertConfig)
        self.myconfig = config
        for param in self.src_embed.parameters():
            param.requires_grad = False

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=bertConfig.hidden_size, out_channels=config.num_channels, kernel_size=config.kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(config.max_sen_len - config.kernel_size[0]+1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=bertConfig.hidden_size, out_channels=config.num_channels, kernel_size=config.kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(config.max_sen_len - config.kernel_size[1]+1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=bertConfig.hidden_size, out_channels=config.num_channels, kernel_size=config.kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(config.max_sen_len - config.kernel_size[2]+1)
        )
        
        for conv in [self.conv1, self.conv2, self.conv3]:
            for m in conv.modules():
                if isinstance(m, nn.Conv1d):  
                    torch.nn.init.xavier_uniform(m.weight)
                    torch.nn.init.constant(m.bias, 0.1)

        self.fc1=torch.nn.Linear(bertConfig.hidden_size, bertConfig.hidden_size)
        self.fc2=torch.nn.Linear(bertConfig.hidden_size, bertConfig.hidden_size)
        self.fc3=torch.nn.Linear(bertConfig.hidden_size, 11)
        self.fc = nn.Linear(config.num_channels*len(config.kernel_size), config.output_size)
        self.dropout = nn.Dropout(config.dropout_keep)
        # h, N, dropout = config.h, config.N, config.dropout
        # d_model, d_ff = config.d_model, config.d_ff
        # # s
        # # Fully-Connected Layer
        # self.fc = nn.Linear(
        #     config.d_model,
        #     config.output_size
        # )
        

        # Softmax non-linearity
        # self.softmax = nn.Softmax(dim=1)
        self.init_weights()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, attention_mask, label):
        output = self.src_embed(x, attention_mask=attention_mask)
        # loss = output.loss
        loss = None
        logits = output.last_hidden_state # shape = (batch_size, sen_len, d_model)
        a = logits.size()
        logits = logits.permute(0, 2, 1)
        # embedded_sent.shape = (batch_size=64,embed_size=300,max_sen_len=20)
        
        conv_out1 = self.conv1(logits).squeeze(2) #shape=(64, num_channels, 1) (squeeze 1)
        conv_out2 = self.conv2(logits).squeeze(2)
        conv_out3 = self.conv3(logits).squeeze(2)
        
        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        # all_out = self.dropout(all_out)
        logits = self.fc(all_out)
        # logits = self.fc1(logits)
        # logits = self.fc2(logits)
        # logits = self.fc3(logits)
        # # encoded_sents = self.encoder(embedded_sents)
        
        # # Convert input to (batch_size, d_model) for linear layer
        # a = x.size()
        # b = embedded_sents.size()
        # final_feature_map = embedded_sents[:,0,:]
        # final_out = self.fc(final_feature_map)
        return self.sigmoid(logits), loss
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        # Exponential
        # self.attenuation = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.gamma)
        
        # Step
        self.attenuation = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.myconfig.max_epochs / 3), gamma=self.myconfig.gamma)
        
    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
                
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
                attention_mask = batch[2].cuda()
            else:
                x = batch[1].type(torch.LongTensor)
                # y = (batch[0] - 1).type(torch.LongTensor)
                y = batch[0]
            y_pred, loss = self.__call__(x, attention_mask, y)
            loss = self.loss_op(y_pred, y)
            
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            if i % 4 == 0:
                print("Iter: {}".format(i+1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []
                
                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model(self, val_iterator)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()
                
        return train_losses, val_accuracies