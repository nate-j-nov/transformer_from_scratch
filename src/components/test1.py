# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022
#
# Test full rough model 

import torch
from Transformer_1 import Transformer_1

def main():
  x = torch.randint(100,(10,)) # source language dummy token seq
  y = torch.randint(100,(12,)) # target language dummy token seq
  print(f"x = {x}")
  print(f"y = {y}")

  model = Transformer_1(512, 100)
  
  out_seq = model(x,y)
  print(f"out_seq: {out_seq}")


if __name__ == "__main__": 
  main()
