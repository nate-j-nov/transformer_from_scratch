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
    '''
    Function to tokenize the text sequences
    Parameters: 
        data_iter: DataIterator (PyTorchType) 
        idx: Index in the DataIterator that indicates which langauge to select the sequence from   
            the langauges come in (lang1, lang2) tuple pairs
    '''
    for pair in data_iter:
        yield tokenizer(pair[idx])

def create_vocabs():
    '''
    Function to create vocabularies. 
    Only creates vocabularies for English and German.
    '''

    # simple function to yield tokens
    # Create vocab and tokenize the German vocab
    de_vocab = build_vocab_from_iterator(yield_tokens(train_iter_de, 0),
                                         specials=["<s>", "</s>", "<blank>","<unk>"])
    de_vocab.set_default_index(de_vocab["<unk>"])
    
    # Create vocab and tokenize the English vocab
    en_vocab = build_vocab_from_iterator(yield_tokens(train_iter_en, 1),
                                         specials=["<s>", "</s>", "<blank>","<unk>"])
    en_vocab.set_default_index(en_vocab["<unk>"])

    return de_vocab, en_vocab

def translate(model, src, max_len, de_vocab, en_vocab, de_pipeline):
    '''
    Function to generate a translated english sentence from a German source string 
    '''
    model.eval()
    # create output tensor, seeded with 0
    # NOTE: I think this can just be zeros
    tgt_tokens = [en_vocab.get_stoi()['<s>']]
    #tgt_tokens = torch.zeros(1, dtype=torch.int64).fill_(en_vocab.get_stoi()['<s>'])
    print(f"Translated tokens: {tgt_tokens}")
    
    # tokenize and add begin/end of sequence tokens to src
    src_bos = torch.zeros(1).fill_(de_vocab.get_stoi()['<s>'])
    src_eos = torch.zeros(1).fill_(de_vocab.get_stoi()['</s>'])
    src_token = torch.cat([src_bos.type(dtype=torch.int64),
                            torch.tensor(de_pipeline(src),dtype=torch.int64),
                            src_eos.type(dtype=torch.int64)
                          ])
    print(f"src token: {src_token}")
    print(f"src token shape: {src_token.shape}")
    
    # encode src using model, store as memory
    with torch.no_grad():
        src_enc = model.encode(src_token)
        print(f"Encoded src shape: {src_enc.shape}")
        
    # loop up to max_len, generate tokens with decoder
    for i in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_tokens)
        #tgt_mask = 
        with torch.no_grad():
            p_toks = model.decode(src_enc, tgt_tensor)
            print(f"decoded token probabilities shape: {p_toks.shape}")
            print(f"decoded token probabilities: {p_toks}")  
            print(f"decoded token last row: {p_toks[-1,:]}")              
        pred = p_toks.argmax(dim=1)[-1]
        print(f"pred: {pred}")
        tgt_tokens.append(int(pred))
        if en_vocab.get_itos()[pred.item()] == en_vocab.get_stoi()['</s>']:
            break
        print(f"target seq: {tgt_tokens}")

def main(argv):
    '''
    Main driver of the application
    '''
    # make main function repeatable by setting a seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.enabled = False

    de_vocab, en_vocab = create_vocabs()

    # prepare the text processing pipeline
    ###
    de_pipeline = lambda x: de_vocab(tokenizer(x))
    en_pipeline = lambda x: en_vocab(tokenizer(x))

    # initialize the network and the optimizer
    network = Transformer_1(512, len(de_vocab), len(en_vocab))
    network.load_state_dict('results/model.pth')

    translate(network, "ich bein ein berliner", 64, de_vocab, en_vocab, de_pipeline)

if __name__ == "__main__":
    main(sys.argv)