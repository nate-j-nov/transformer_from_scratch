# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022

import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module): 
    ''' 
    Class definition for a feed forward network: 
    Documentation help: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    '''

    def __init__(self, d_model, d_hidden):
        super(FeedForwardNetwork, self).__init__(); 
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
    
    def forward(self, x): 
        l1 = self.linear1(x)
        relu = F.relu(l1); 
        return self.linear2(relu); 
