# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022
#
# Test various modules 

import torch
from EncoderModule import EncoderModule
from DecoderModule import DecoderModule

def main():
  x = torch.ones(10,512) # source language dummy token seq
  y = torch.randn(12,512) # target language dummy token seq
  print(f"Tensor x: {x}")
  print(f"Tensor y: {y}")

  # encode x tensor of size 10,512
  new_encoder = EncoderModule()
  x1 = new_encoder(x)
  print(f"Encoder output: {x1}")

  # decode y tensor of size 12,512 with encoder module output x1
  new_decoder = DecoderModule(x1)
  y1 = new_decoder(y)
  print(f"Decoder output: {y1}")


if __name__ == "__main__": 
    main()
