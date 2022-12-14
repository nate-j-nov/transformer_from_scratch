# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022
#
# This program contains functions that load an English-German machine translation dataset
#   and defines train and test functions

# following torchtext tutorial at 
# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html


# import statements
import sys
import math
import os

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchdata
import torchtext.datasets as datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Transformer_1 import Transformer_1

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

# global variables
RANDOM_SEED = 1 # random seed value

# create training dataset iterator from Multi30k train
train_iter_en = iter(datasets.Multi30k(split='train', language_pair=('de','en')))
train_iter_de = iter(datasets.Multi30k(split='train', language_pair=('de','en')))
tokenizer = get_tokenizer('basic_english')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def yield_tokens(data_iter, idx):
    for pair in data_iter:
        yield tokenizer(pair[idx])

def create_vocabs():
    '''
        Create German and English vocabularies
    '''
    # simple function to yield tokens
    de_vocab = build_vocab_from_iterator(yield_tokens(train_iter_de, 0),
                                         specials=["<s>", "</s>", "<blank>","<unk>"])
    de_vocab.set_default_index(de_vocab["<unk>"])
    
    en_vocab = build_vocab_from_iterator(yield_tokens(train_iter_en, 1),
                                         specials=["<s>", "</s>", "<blank>","<unk>"])
    en_vocab.set_default_index(en_vocab["<unk>"])
    #display("german test: ", de_vocab(['ein', 'in', 'bedienen', 'ich', 'vieler']))
    #display("english test: ", en_vocab(['man', 'is', 'a', 'blue', 'elephant']))
    return de_vocab, en_vocab

# TODO: move this to main

def collate_batch(batch, de_pipeline, en_pipeline, max_padding=128, pad_id=2):
    bos = torch.tensor([0], dtype=torch.int64)
    eos = torch.tensor([1], dtype=torch.int64)
    de_list, en_list = [], []
    for (_de, _en) in batch:
        #display(_de, _en)
        # process german
        de_processed = torch.tensor(de_pipeline(_de), dtype=torch.int64)
        de_processed = torch.cat([bos, de_processed, eos], dim=0)
        # TODO: uncomment if batching
        #de_processed = F.pad(de_processed, (0, max_padding - de_processed.size(0)), value=pad_id)
        de_list.append(de_processed)
        
        # process english
        en_processed = torch.tensor(en_pipeline(_en), dtype=torch.int64)
        en_processed = torch.cat([bos, en_processed, eos], dim=0)
        # TODO: uncomment if batching
        #en_processed = F.pad(en_processed, (0, max_padding - en_processed.size(0)), value=pad_id)
        en_list.append(en_processed)
    de_list = torch.cat(de_list) 
    en_list = torch.cat(en_list)
    return de_list.to(device), en_list.to(device)


# useful functions and helper functions
''' Train the network given the passed parameters

Arguments:
network -- the neural network to be trained
optimizer -- the optimizer object used to train weights
train_loader -- a DataLoader of training data
train_losses -- the calculated loss value after each data point is used for training
train_counter -- a list storing the number of training data points used
epoch -- the current epoch being trained
log_interval -- the interval at which to write data to the log
'''
# https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
def train_model( model, optimizer, train_loader, train_losses, train_counter, 
        epoch, log_interval, criterion ):
    model.train()
    # data - English sentence, German sentence, mask:
    for batch_idx, (src, tgt) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output, tgt)
        loss.backward() # calculate gradients
        optimizer.step() # adjust learning weights
        
        # Log interval
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} | Progress: {batch_idx} | Loss {loss.item()}")
            torch.save(model.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')


#            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                epoch, batch_idx * len(src), len(train_loader.dataset),
#                100. * batch_idx / len(train_loader), loss.item()))
#            train_losses.append(loss.item())
#            train_counter.append(
#                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
#            torch.save(model.state_dict(), './results/model.pth')
#            torch.save(optimizer.state_dict(), './results/optimizer.pth')

def eval_model(model, criterion, eval_loader): 
    model.eval()
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(eval_loader):
            output = model(src, tgt)
            loss = criterion(output, tgt)
            print(f"Loss: {loss}")
            
            ## Log interval
            #if batch_idx % log_interval == 0:
            #    print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            #        epoch, batch_idx * len(src), len(train_loader.dataset),
            #        100. * batch_idx / len(train_loader), loss.item()))
            #    train_losses.append(loss.item())
            #    train_counter.append(
            #        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            #    torch.save(model.state_dict(), './results/model.pth')
            #    torch.save(optimizer.state_dict(), './results/optimizer.pth')

def main(argv):

    # make main function repeatable by setting a seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.enabled = False

    # set hyperparameters
    epochs = 1
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    de_vocab, en_vocab = create_vocabs()
    
    # prepare the text processing pipeline
    ###
    de_pipeline = lambda x: de_vocab(tokenizer(x))
    en_pipeline = lambda x: en_vocab(tokenizer(x))

    def collate_fn(batch): 
        return collate_batch(batch, de_pipeline, en_pipeline)

    # initialize the network and the optimizer
    network = Transformer_1(512, len(de_vocab), len(en_vocab))

    optimizer = optim.Adam(network.parameters(), betas=(0.9, 0.98), eps=10e-9)

    train_iter, eval_iter = datasets.Multi30k(split=('train', 'valid'))

    train_loader = DataLoader(train_iter, batch_size=1, shuffle=False, collate_fn=collate_fn)
    valid_loader = DataLoader(eval_iter, batch_size=1, shuffle=False, collate_fn=collate_fn)

    #for idx, (de_t, de_off, en_t, en_off) in enumerate(dataloader):


    # display("de pipeline: ", de_pipeline('ich bin ein berliner'))
    # display("en pipeline: ", en_pipeline('here is the english example'))

    ### generate data batch and iterator

    criterion = nn.CrossEntropyLoss(); 

    ###
    # initialize lists for tracking performance
    train_losses = []
    train_counter = []
    test_losses = []
#    test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]
    #test_counter = [i*len(data[0].dataset) for i in range(epochs + 1)]

    #test( network, test_loader, test_losses )
    for epoch in range(1, epochs + 1):
        #train_network( network, optimizer, train_loader, epoch, log_interval )
        train_model( network, optimizer, train_loader, train_losses, train_counter, 
            epoch, log_interval, criterion )
        #test( network, test_loader, test_losses )
        eval_model( network, valid_loader, test_losses )


if __name__ == "__main__":
    main(sys.argv)