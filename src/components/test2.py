# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022
#
# Test full rough model 

import seaborn as sns
import torch
from Transformer_1 import Transformer_1

def main():
  x = torch.randint(100,(10,)) # source language dummy token seq
  y = torch.randint(100,(12,)) # target language dummy token seq
  print(f"x = {x}")
  print(f"y = {y}")

  # src vocab: de_vocab size = 18_757
  # tgt vocab: en_vocab size = 10_210
  model = Transformer_1(512, 18757, 10210)
  model.load_state_dict(
        torch.load('results/model.pth', map_location=torch.device("cpu"))
  )


#  Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.
  test_src = torch.tensor([0,18,26,251,30,84,20,88,7,15,110, 7605,3160,4,1])
  print(f"test_src: {test_src}")

  print(f"model embedding: {model.embedding}")
  x = model.embedding(x)
  print(f"embedded: {x}")

  print(f"model pos_enc: {model.pos_enc}")
  x = model.pos_enc(x)
  print(f"encoded: {x}")

  print(f"model encoder mha: {model.encoder.mha}")
  x, attn_map = model.encoder.mha(x,x,x)
  print(f"attended to: {x}")
  print(f"attn_map shape: {attn_map.shape}")
  print(f"attention map: {attn_map}")

  sns.heatmap(attn_map.detach().numpy())



  #out_seq = model(x,y)
  #print(f"out_seq size: {out_seq.shape}")
  #print(f"out_seq: {out_seq}")


if __name__ == "__main__": 
  main()
