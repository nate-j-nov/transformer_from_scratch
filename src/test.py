# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022
#
# Test various modules 

import torch
from EncoderModule import EncoderModule
from DecoderModule import DecoderModule
from PositionalEncoder import PositionalEncoder

def main():
  x = torch.zeros(10,512) # source language dummy token seq
  y = torch.randn(12,512) # target language dummy token seq
  print(f"x datatype: {x.dtype}")
  print(f"Tensor x: {x}")
  print(f"Tensor y: {y}")

  # apply positional encoding to dummy input token seq
  new_pos_enc = PositionalEncoder(x.size()[1])
  x_pos_enc = new_pos_enc(x)
  print(f"Positionally encoded size: {x_pos_enc.size()}")
  print(f"Positionally encoded tensor: {x_pos_enc}")

  # encode x tensor of size 10,512
  new_encoder = EncoderModule(512,8) # encoder with d_model 512, 8-head MHA, drop=0.1 by default
  x1 = new_encoder(x_pos_enc)
  print(f"Encoder output size: {x1.size()}")
  print(f"Encoder output: {x1}")

  # decode y tensor of size 12,512 with encoder module output x1
  new_decoder = DecoderModule(512,8)
  y1 = new_decoder(y, x1)
  print(f"Decoder output size: {y1.size()}")
  print(f"Decoder output: {y1}")


if __name__ == "__main__": 
    main()
