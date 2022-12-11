# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022
#
# Transformer_1.py contains the architecture and forward function for a basic implementation
#   of a transformer model, including input/output embedding, positional encoding,
#   one encoder module, one decoder module, and an output linear layer with softmax.

import torch
import torch.nn as nn
from PositionalEncoder import PositionalEncoder
from EncoderModule.py import EncoderModule
from DecoderModule.py import DecoderModule

###
# TODO: tie weights on input embedding and output embedding using following example
# https://github.com/pytorch/examples/blob/main/word_language_model/model.py#L28
###


class Transformer_1(nn.Module):
  '''
    Transformer model as described in "Attention Is All You Need" (2017) by
    Vaswani, et al. 
    This is an Encoder-Decoder model that uses the Encoder and Decoder modules
    described in EncoderModule.py and DecoderModule.py. This version of the 
    transformer model uses N=1 copies of both the encoder and decoder, as
    opposed to the N=6 used in the original Vaswani paper.
  '''

    def __init__(self, d_model, vocab_size):
      super(Transformer_1, self).__init__()
      self.embedding = nn.Embedding(d_model, vocab_size)
      self.encoder = EncoderModule(d_model, 8, 0.1) # encoder with 8-head MHA and dropout 0.1
      self.decoder = 
### decoder module
### this is probably not right - how to pass encoder output to decoder init
###   when encoder hasn't been run yet? Empty tensor of proper size?A
### TODO: make size of model a parameter rather than hard-coded
      self.encoder_out = torch.ones(2, 512)
      self.decoder = DecoderModule(encoder_out)
