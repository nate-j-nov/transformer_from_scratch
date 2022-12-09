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

    def __init__(self):
        super(FeedForwardNetwork, self).__init__(); 
        self.linear1 = nn.Linear(512, 2048)
        self.linear2 = nn.Linear(2048, 512)
    
    def forward(self, x): 
        l1 = self.linear1(x)
        relu = F.relu(l1); 
        return self.linear2(relu); 
