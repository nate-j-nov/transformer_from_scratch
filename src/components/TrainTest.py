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

#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torch.utils.data import Dataset
#from torch.utils.data import DataLoader

#import torchvision
#from torchvision.transforms import ToTensor, Lambda
#from torchvision.io import read_image
#import torchvision.models as models
##from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# global variables
RANDOM_SEED = 1 # random seed value


# create training dataset iterator from Multi30k train
train_iter_en = iter(datasets.Multi30k(split='train', language_pair=('de','en')))
train_iter_de = iter(datasets.Multi30k(split='train', language_pair=('de','en')))
tokenizer = get_tokenizer('basic_english')



# create vocabularies
def create_vocabs():
  '''
    Create German and English vocabularies
  '''
    # simple function to yield tokens
    def yield_tokens(data_iter, idx):
        for pair in data_iter:
            yield tokenizer(pair[idx])
    de_vocab = build_vocab_from_iterator(yield_tokens(train_iter_de, 0), 
                                         specials=["<s>", "</s>", "<blank>","<unk>"])                                     
    de_vocab.set_default_index(vocab["<unk>"])
    
    en_vocab = build_vocab_from_iterator(yield_tokens(train_iter_en, 1), 
                                         specials=["<s>", "</s>", "<blank>","<unk>"])
    en_vocab.set_default_index(vocab["<unk>"])
    #display("german test: ", de_vocab(['ein', 'in', 'bedienen', 'ich', 'vieler']))
    #display("english test: ", en_vocab(['man', 'is', 'a', 'blue', 'elephant']))
    return de_vocab, en_vocab

# TODO: move this to main
de_vocab, en_vocab = create_vocabs()

###
# prepare the text processing pipeline
###
de_pipeline = lambda x: de_vocab(tokenizer(x))
en_pipeline = lambda x: en_vocab(tokenizer(x))

# display("de pipeline: ", de_pipeline('ich bin ein berliner'))
# display("en pipeline: ", en_pipeline('here is the english example'))

### generate data batch and iterator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch, max_padding=16, pad_id=2):
    bos = torch.tensor([0], dtype=torch.int64)
    eos = torch.tensor([1], dtype=torch.int64)
    de_list, en_list = [], []
    for (_de, _en) in batch:
        display(_de, _en)
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
    de_list = torch.stack(de_list)
    en_list = torch.stack(en_list)
    return de_list.to(device), en_list.to(device)

train_iter = datasets.Multi30k(split='train')

dataloader = DataLoader(train_iter, batch_size=1, shuffle=False, collate_fn=collate_batch)

i=0

#for idx, (de_t, de_off, en_t, en_off) in enumerate(dataloader):
for idx, (de_t, en_t) in enumerate(dataloader):
    i+=1
    if i == 2:
        break
    print(de_t)
    print(f"en_t: {en_t}")
    print(f"en_t shape: {en_t.shape}")














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
        epoch, log_interval ):
    model.train()
# data - English sentence, German sentence, mask:
    for batch_idx, (src, tgt) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output, target)
        loss.backward() # calculate gradients
        optimizer.step() # adjust learning weights
        # if batch_idx % log_interval == 0:
        #     print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        #     train_losses.append(loss.item())
        #     train_counter.append(
        #         (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        #     torch.save(model.state_dict(), './results/model.pth')
        #     torch.save(optimizer.state_dict(), './results/optimizer.pth')

''' Load MNIST data, train and test neural network, and plot results '''
def main(argv):
    # handle any command line arguments in argv

    # make main function repeatable by setting a seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.enabled = False

    # set hyperparameters
    epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    # main function code
    # import MNIST data and store in DataLoaders
    data = import_Multi30k( batch_size_train, batch_size_test ) 
    # plot first 8 examples
    plot_k( data[0], 6 )

    # initialize the network and the optimizer
    network = MNIST_Network()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # initialize lists for tracking performance
    train_losses = []
    train_counter = []
    test_losses = []
#    test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]
    test_counter = [i*len(data[0].dataset) for i in range(epochs + 1)]

    #test( network, test_loader, test_losses )
    test_network( network, data[1], test_losses )
    for epoch in range(1, epochs + 1):
        #train_network( network, optimizer, train_loader, epoch, log_interval )
        train_network( network, optimizer, data[0], train_losses, train_counter, 
            epoch, log_interval )
        #test( network, test_loader, test_losses )
        test_network( network, data[1], test_losses )

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig
    plt.show()
    return


if __name__ == "__main__":
    main(sys.argv)


  
  
  
  
''' Test the network given the passed parameters

Arguments:
network -- the neural network to be tested 
test_loader -- a DataLoader of test data
test_losses -- the calculated loss value after each data point is used for testing 
test_counter -- a list storing the number of test data points used
'''
# def test_network( network, test_loader, test_losses ):
#     network.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = network(data)
#             test_loss += F.nll_loss(output, target, size_average=False).item()
#             pred = output.data.max(1, keepdim=True)[1]
#             correct += pred.eq(target.data.view_as(pred)).sum()
#         test_loss /= len(test_loader.dataset)
#         test_losses.append(test_loss)
#         print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#             test_loss, correct, len(test_loader.dataset), 
#             100. * correct / len(test_loader.dataset)))





