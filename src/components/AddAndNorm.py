# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022
# Class definition AddAndNorm class that takes two tensors, sums them, 
#   and takes the LayerNorm of the sum

import torch.nn as nn

class AddAndNorm(nn.Module): 
  '''
  Add and norm
  '''

  def __init__(self):
    super(AddAndNorm, self).__init__()
    self.layer_norm = nn.LayerNorm(512);
    
  def forward(self, orig_input, sublayer_output): 
    if orig_input.size != sublayer_output.size:
      raise InputError("Tensors must be the same shape to be added and normed")
    tensor_sum = orig_input + sublayer_output
    return self.layer_norm(tensor_sum)
