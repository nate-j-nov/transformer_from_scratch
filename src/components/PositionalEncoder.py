# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022
#
# This file contains the class definition for a Positional Encoding module

import numpy as np
import torch
import torch.nn as nn

class PositionalEncoder(nn.Module): 
  '''
    From "Attention Is All You Need" by Vaswani, et al.:
    " In order for the model to make use of the order of the sequence, we must inject some 
    information about the relative or absolute position of the tokens in the sequence. To that end,
    we add 'positional encodings' to the input embeddings at the bottoms of the encoder and decoder
    stacks. The positional encodings have the same dimension d_model as the embeddings, so the two
    can be summed."
    We follow Vaswani, et al.'s example and use sine and cosine functions of different frequences,
    given by the following formula:
    PE(pos, 2i) = sin( pos / (10,000 ^ (2i/d_model)) )
    PE(pos, 2i+1) = cos( pos / (10,000 ^ (2i/d_model)) )
    where i is in range [0, d_model-1]
  '''

  def __init__(self, d_model, dropout=0.1):
    super(PositionalEncoder, self).__init__()
    self.d_model = d_model 
    self.dropout = nn.Dropout(p=dropout)

  # compute a forward pass of the positional encoder
  def forward(self, x): 
    seq_len = x.size()[0]
    seq_dim = x.size()[1]
    if seq_dim != self.d_model:
      raise ValueError(f"Input dimension {seq_dim} does not match positional encoder dimension {self.d_model}") 
    pe = np.arange(seq_len*self.d_model)*1.0
    pe = np.reshape(pe, [seq_len, seq_dim])
    pos_array = pe // seq_dim
    dim_array = pe % seq_dim
    # positional encoding calculation for even dimensions
    pe[:,::2] = np.sin(pos_array[:,::2] / np.power(10_000, (dim_array[:,::2]/seq_dim)))
    pe[:,1::2] = np.cos(pos_array[:,1::2] / np.power(10_000, ((dim_array[:,1::2]-1)/seq_dim)))
    positions = torch.from_numpy(pe)
    # cast to float32 so that mha in EncoderModule doesn't throw an error
    # TODO: why do I need to cast this? 
    positions = positions.to(torch.float32)
    # note that dropout applies a scaling factor of 1/(1-p) to the non-dropped values
    #   to keep expected value of weights the same - this means that the outputs
    #   from this layer look weird (e.g. 0+1=1.1111 i.e. 1/0.9)
    return self.dropout(x + positions) # apply dropout to sum of input and encoding tensor 

