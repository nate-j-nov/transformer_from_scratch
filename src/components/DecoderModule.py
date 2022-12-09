# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022
#
# This file contains the class definition for a Decoder module

import torch
import torch.nn as nn
from FeedForward import FeedForwardNetwork 
from AddAndNorm import AddAndNorm 
from EncoderModule import EncoderModule

class DecoderModule(nn.Module): 
  '''
    Decode the encoded sequence of continuous representations z=(z1, ..., zn)
  '''

  def __init__(self, encoder_output):
    super(DecoderModule, self).__init__()
    self.mha = nn.MultiheadAttention(512, 8) # multihead attention with 8 head
    self.ff = FeedForwardNetwork() # feed forward network described in FeedForward.py
    self.layer_norm = nn.LayerNorm(512) # layer norm with dimension d_model = 512
    self.drop = nn.Dropout(p=0.1) # dropout layer with p=0.1 for regularization
    self.encoder_output = encoder_output # final output of encoder model (i.e. z)
    # self.addnorm = AddAndNorm()

  # compute a forward pass of a decoder module 
  def forward(self, x): 
    # mask tensor for masked multi-head attention - upper triangular ones matrix same size as x
    tgt_seq_len = x.size()[0]
    mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1) 
    # each step of forward pass takes LayerNorm of a residual connection, followed by dropout
    if self.mha(x,x,x,attn_mask=mask)[0].size() != x.size():
      print(f"mha out size: {self.mha(x,x,x,attn_mask=mask)[0].size()}")
      raise ValueError("Multihead attn output should be same size as input")
    x = self.drop(self.layer_norm(x + self.mha(x,x,x,attn_mask=mask)[0])) # multihead attention with q=k=v
    x = self.drop(self.layer_norm(x + self.mha(x,self.encoder_output, self.encoder_output)[0]))
    x = self.drop(self.layer_norm(x + self.ff(x))) # feed-forward network
    return x
