# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022
#
# Transformer_1.py contains the architecture and forward function for a basic implementation
#   of a transformer model, including input/output embedding, positional encoding,
#   one encoder module, one decoder module, and an output linear layer with softmax.

import torch
import torch.nn as nn
import torch.nn.functional as F
from PositionalEncoder import PositionalEncoder
from EncoderModule import EncoderModule
from DecoderModule import DecoderModule

###
# TODO: tie weights on input embedding and output embedding using following example
# https://github.com/pytorch/examples/blob/main/word_language_model/model.py#L28
# TODO: embedding layers needs to be multiplied by sqrt(d_model)
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

  def __init__(self, d_model, src_vocab_size, tgt_vocab_size, tie_weights=False):
    super(Transformer_1, self).__init__()
    self.embedding = nn.Embedding(src_vocab_size, d_model) # encoder and decoder input embeddings
    self.pos_enc = PositionalEncoder(d_model, 0.1) # positional encoder with droppout 0.1
    self.encoder = EncoderModule(d_model, 8, 0.1) # encoder with 8-head MHA and dropout 0.1
    self.decoder = DecoderModule(d_model, 8, 0.1) # decoder with 8-head MHA and dropout 0.1
    self.out_embed = nn.Linear(d_model, tgt_vocab_size) # linear layer predict next token

    # weight tying example from PyTorch:
    # NOTE: src_vocab_size must equal tgt_vocab_size if tie_weights is true
    # https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    if tie_weights == True:
      self.embedding.weight = self.out_embed.weight

  def forward(self, x, y):
    # process input sequence
    inp = self.embedding(x) # embed tokenized sequence
    inp = self.pos_enc(inp) # add positional encoding
    inp = self.encoder(inp) # apply encoder module

    # process output sequence
    outp = self.embedding(y)
    outp = self.pos_enc(outp)
    outp = self.decoder(outp, inp)
    
    # final steps
# TODO: figure out if the dim=1 in the softmax is correct
    p_token = F.softmax(self.out_embed(outp), dim=1)
    return p_token
      

