# train.py

from utils import *
from model import *
from config import Config
import sys
import torch.optim as optim
from torch import nn
import torch

if __name__=='__main__':
    config = Config()
    train_file = './data/CVE2CWE/cve.train'
    if len(sys.argv) > 2:
        train_file = sys.argv[1]
    test_file = './data/CVE2CWE/cve.test'
    if len(sys.argv) > 3:
        test_file = sys.argv[2]
    
    dataset = Dataset(config)
    dataset.load_data(train_file, test_file)
    
    # Create Model with specified optimizer and loss function
    ##############################################################
    model = Transformer(config, len(dataset.vocab))
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0002)
    loss_func = nn.BCELoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(loss_func)
    ##############################################################
    
    train_losses = []
    val_accuracies = []
    best = 0
    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        print("\t Learning Rate: {:.5f}".format(model.optimizer.state_dict()['param_groups'][0]['lr']))
        train_loss,val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        model.attenuation.step()
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        
    print("##########FINAL RESULTS##########")
    train_acc = model.scorer.evaluate_model(model, dataset.train_iterator, "Train")
    print("#################################")
    val_acc = model.scorer.evaluate_model(model, dataset.val_iterator, "Validation")
    print("#################################")
    test_acc = model.scorer.evaluate_model(model, dataset.test_iterator, "Test")
    print("#################################")
    save_model(model, 'ckpts/transformer-cve.pkl')
    # save_model(dataset.vocab, 'ckpts/vocab.pkl')
