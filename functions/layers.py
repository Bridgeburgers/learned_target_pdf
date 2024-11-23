# -*- coding: utf-8 -*-
"""
Custom torch layers for learned-pdf 
"""

import torch
import torch.nn as nn
import numbers

class learned_pdf_mc_mlp(nn.Module):
    def __init__(self, vector_dim, hidden_dims, target_range, device, 
                 mc_samples=10, delta_t=0.01, mlp_activation=nn.ReLU(), 
                 dropout=0.1, torch_seed=123):
        super().__init__()
        
        torch.random.seed(torch_seed)
        
        self.t_range = torch.arange(target_range[0], target_range[1]+delta_t, delta_t).to(device)
        self.mc_samples = mc_samples
        self.target_range = target_range
        self.device = device
        
        if isinstance(hidden_dims, numbers.Number):
            hidden_dims = [hidden_dims]
        
        hidden_dims = [vector_dim + 1] + hidden_dims
        mlp_layers = [
            layer
            for prev_num, num in zip(hidden_dims[0:-1], hidden_dims[1:])
            for layer in [nn.Linear(prev_num, num), nn.Dropout(dropout), mlp_activation]
            ]
        self.mlp = nn.Sequential(*mlp_layers)
        self.y_layer = nn.Linear(hidden_dims[-1], 1)
        
    def y_multiple_t_samples(self, x, t_array):
        #reshape t_array from [len(t_array)] to [x.shape[0], len(t_array), 1] repeating it along the first dimension
        t_array_repeat = t_array.expand(x.shape[0], len(t_array))[:,:,None]
        
        #reshape x from [x.shape[0], x.shape[1]] to 
        #[x.shape[0], len(t_array), x.shape[1] repeating it along the second dimension]
        x_reshape = x.unsqueeze(1).expand(x.shape[0], t_array_repeat.shape[1], x.shape[1])
        
        #add t_array_repeat as another "feature" to x along the third dimension
        x_t_array = torch.cat([x_reshape, t_array_repeat], axis=2)
        
        z_array = self.mlp(x_t_array)
        y_array = self.y_layer(z_array).squeeze(-1)
        
        return y_array
    
    def y_single_t_sample(self, x, t):
        x_t = torch.cat([x, t.reshape(-1,1)], axis=1)
        z = self.mlp(x_t)
        y = self.y_layer(z).squeeze(-1)
        
        return y

    def forward(self, x, t):
        y = self.y_single_t_sample(x,t)

        #create uniform samples in target range
        t_random_array = torch.rand(self.mc_samples) *\
            (self.target_range[1] - self.target_range[0]) + self.target_range[0]
        t_random_array = t_random_array.to(self.device)
        y_array = self.y_multiple_t_samples(x, t_random_array)
        
        return y, y_array
    
    def integration_forward(self, x, t):
        y = self.y_single_t_sample(x,t)
        y_range = self.y_multiple_t_samples(x, self.y_range)
        
        return y, y_range


class learned_pdf_integration_mlp(learned_pdf_mc_mlp):
    
    def forward(self, x, t):
        return super().integration_forward(x, t)