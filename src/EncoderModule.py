# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022
#
# This file contains the class definition for an Encoder module

import torch.nn as nn
from FeedForward import FeedForwardNetwork 

class EncoderModule(nn.Module): 
    '''
    An Encoder "layer" from "Attention Is All You Need" by Vaswani et al. We are referring
    to it as a "module" because Vaswani et al. use the term "layer" or "sub-layers".
    Includes a multi-head attention sub-layer and feed-forward sub-layer, both of which
    have LayerNorm and residual connections applied.

    Inherits from nn.Module

    Conducts: 
        Multihead attention (on the positionally encoded input embeddings) -> Add and Norm 
        Feed Forward (on outuput from MHA) -> Add and Norm
    '''

    def __init__(self, d_model, num_heads, dropout=0.1):
        '''
        Constructor for the Encoder Module
        Parameters: 
            d_model: Model dimension size 
            num_heads: number of heads for multi-head attention
            drop_out: drop out rate for training. Default: 0.1
        '''
        super(EncoderModule, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads) # multihead attention 
        self.ff = FeedForwardNetwork(d_model, 2048) # feed forward network described in FeedForward.py
        self.layer_norm = nn.LayerNorm(d_model) # layer norm with dimension d_model = 512
        self.drop = nn.Dropout(p=dropout) # dropout layer with p=0.1 for regularization
    
    # compute a forward pass of an encoder module 
    def forward(self, x): 
        '''
        Forward fuction for encoder module
        Parameters: 
            x: input tensor representing either the encoded input 
        '''
        # each step of forward pass takes LayerNorm of a residual connection, followed by dropout
        if self.mha(x,x,x)[0].size() != x.size():
            raise ValueError("Multihead attn output should be same size as input")
        x = self.drop(self.layer_norm(x + self.mha(x,x,x)[0])) # multihead attention with q=k=v
        x = self.drop(self.layer_norm(x + self.ff(x))) # feed-forward network
        return x