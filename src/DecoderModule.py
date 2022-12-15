# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022
#
# This file contains the class definition for a Decoder module

import torch
import torch.nn as nn
from FeedForward import FeedForwardNetwork 
from EncoderModule import EncoderModule

class DecoderModule(nn.Module): 
  '''
    Decode the encoded sequence of continuous representations z=(z1, ..., zn)
    Inherits from nn.Module. 
    Conducts: 
        Masked Multi-Head Attention on positionally encoded output embedding -> Add and Norm
        Multi-Head Attention on MMHA Output and output from the encoder -> Add and Norm
        Feed Forward Network -> Add and Norm
  '''

  def __init__(self, d_model, num_heads, dropout=0.1) -> None: 
    '''
    Constructor for the Decoder Module
    Parameters: 
        d_model: Model dimension size 
        num_heads: number of heads for multi-head attention
        drop_out: drop out rate for training. Default: 0.1
    '''
    super(DecoderModule, self).__init__()
    self.mha = nn.MultiheadAttention(d_model, num_heads) # multihead attention with 8 head
    self.ff = FeedForwardNetwork(d_model, 2048) # feed forward network described in FeedForward.py
    self.layer_norm = nn.LayerNorm(d_model) # layer norm with dimension d_model = 512
    self.drop = nn.Dropout(p=dropout) # dropout layer with p=0.1 for regularization

  # compute a forward pass of a decoder module 
  def forward(self, x, encoder_output): 
    '''
    Forward for decoder function
    Parameters: 
        x: input tensor representing either the encoded input 
        encoder_output: tensor that is the encoded d_model dimensional vectors of the input 
    '''
    # mask tensor for masked multi-head attention - upper triangular ones matrix same size as x
    tgt_seq_len = x.size()[0]
    mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1) 
    # check that dimensions and sizes match
    if self.mha(x,x,x,attn_mask=mask)[0].size() != x.size(): 
      print(f"mha out size: {self.mha(x,x,x,attn_mask=mask)[0].size()}")
      raise ValueError("Multihead attn output should be same size as input")
    # each step of forward pass takes LayerNorm of a residual connection, followed by dropout
    x = self.drop(self.layer_norm(x + self.mha(x,x,x,attn_mask=mask)[0])) # multihead attention with q=k=v
    x = self.drop(self.layer_norm(x + self.mha(x,encoder_output, encoder_output)[0]))
    x = self.drop(self.layer_norm(x + self.ff(x))) # feed-forward network
    return x
