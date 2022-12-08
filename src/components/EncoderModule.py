# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022
#
# This file contains the class definition for an Encoder module

import torch.nn as nn
from FeedForward import FeedForwardNetwork 
from AddAndNorm import AddAndNorm 

class EncoderModule(nn.Module): 
  '''
  Add and norm
  '''

  def __init__(self):
    super(EncoderModule, self).__init__()
    self.mha = nn.MultiheadAttention(512, 8) # multihead attention with 8 head
    self.ff = FeedForwardNetwork() # feed forward network described in FeedForward.py
    self.layer_norm = nn.LayerNorm(512) # layer norm with dimension d_model = 512
    self.drop = nn.Dropout(p=0.1) # dropout layer with p=0.1 for regularization
    # self.addnorm = AddAndNorm()
   
  # compute a forward pass of an encoder module 
  def forward(self, x): 
    # each step of forward pass takes LayerNorm of a residual connection, followed by dropout
    if self.mha(x).size != x.size:
      raise InputError("Multihead attn output should be same size as input")
    x = self.drop(self.layer_norm(x + self.mha(x,x,x))) # multihead attention with q=k=v
    x = self.drop(self.layer_norm(x + self.ff(x))) # feed-forward network
    return x
#    x = self.drop(self.addnorm(x, self.mha(x))) # apply multihead attention 
#    x = self.drop(self.addnorm(x, self.ff(x))) # apply feed-forward network

