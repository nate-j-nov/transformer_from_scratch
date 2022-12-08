# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022

import torch.nn as nn

class AddAndNorm(nn.Module): 
  '''
  Add and norm
  '''

  def __init__(self):
    super(AddAndNorm, self).__init__()
    self.layer_norm = nn.LayerNorm(512);
    
  def forward(self, x): 
    return self.layer_norm(x)