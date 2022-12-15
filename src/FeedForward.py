# Brenden Collins // Nate Novak
# CS7180 Advanced Perception
# Fall 2022

import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module): 
    ''' 
    Class definition for a feed forward network as described in the Vaswani, et. al. paper: 

    Documentation help: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    
    Conducts: 
        Linear layer expanding up from d_model up to d_hidden (takes the output from mulithead attention) (in the case of Vaswani et al., it's 512 -> 2048) -> ReLU
        Another linear layer that takes expanded model and reduces it back to d_model (in the case of Vas. et. al. it's 2048 -> 512
    '''
    def __init__(self, d_model, d_hidden):
        '''
        Constructor for the Feed Forward Network 
        Parameters: 
            d_model: Dimensionality of model
            d_hidden: Size to expand to in the linear layer
        '''
        super(FeedForwardNetwork, self).__init__(); 
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
    
    def forward(self, x): 
        '''
        Forward function for the FeedForwardNeetwork
        Parameters: 
            x: input tensor of the results from multi-head attention either in the encoder or decoder layers
        '''
        l1 = self.linear1(x) # Expand from 512 -> 2048
        relu = F.relu(l1); # ReLU 
        return self.linear2(relu); # Contraction from 2048 -> 512