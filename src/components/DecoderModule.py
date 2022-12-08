# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022
#
# This file contains the class definition for a Decoder module

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
    mask = torch.triu(torch.ones(x.size())) 
    # each step of forward pass takes LayerNorm of a residual connection, followed by dropout
    if self.mha(x).size != x.size:
      raise InputError("Multihead attn output should be same size as input")
    x = self.drop(self.layer_norm(x + self.mha(x,x,x,attn_mask=mask))) # multihead attention with q=k=v
    x = self.drop(self.layer_norm(x + self.mha(x,self.encoder_output, self.encoder_output)))
    x = self.drop(self.layer_norm(x + self.ff(x))) # feed-forward network
    return x
