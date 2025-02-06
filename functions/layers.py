# -*- coding: utf-8 -*-
"""
Custom torch layers for learned-pdf 
"""

import torch
import torch.nn as nn
from torch.distributions.gamma import Gamma
import numbers

class Learned_Pdf_MC_MLP(nn.Module):
    def __init__(self, vector_dim, hidden_dims, target_range, device, 
                 mc_samples=10, delta_t=torch.tensor(0.01), 
                 mlp_activation=nn.ReLU(), dropout=0.1, torch_seed=123):
        '''
        Layer that mixes real-valed target with input vector, to generate a
        function that depends on both inputs and target via dense mlp. This 
        produces the raw output that can be turned into a probability 
        distribution of the target with a softdelta activation. It also outputs 
        MC-sampled y-values (untransformed distribution values) for the MC 
        learned-target loss function

        Parameters
        ----------
        vector_dim : int
            dimension of input vector (can be features, or latent feature space) 
        hidden_dims : list[int]
            list of hidden dimensions in mlp that mixes vector and target
        target_range : list[float]
            2-element list describing the range of the target
        device : torch.device
            torch device (converts output vectors to this device)
        mc_samples : int, optional
            Number of monte-carlo integration samples The default is 10.
        delta_t : float, optional
            delta step for numeric integration if using numeric integration 
            loss function, or just for outputting the entire distribution
            during inference. The default is torch.tensor(0.01).
        mlp_activation : torch.nn.modules.Module, optional
            activation of each hidden layer in the MLP. 
            The default is nn.ReLU().
        dropout : float, optional
            Dropout used for each hidden layer in MLP. The default is 0.1.
        torch_seed : int, optional
            Torch seed value. The default is 123.

        Returns
        -------
        None.

        '''
        super().__init__()
        
        torch.manual_seed(torch_seed)
        
        self.t_range = torch.arange(target_range[0], target_range[1]+delta_t, delta_t).to(device)
        self.mc_samples = mc_samples
        self.device = device
        
        if not torch.is_tensor(target_range):
            target_range = torch.tensor(target_range)
        self.target_range = target_range
        
        if isinstance(hidden_dims, numbers.Number):
            hidden_dims = [hidden_dims]
            
        if isinstance(delta_t, numbers.Number):
            delta_t = torch.tensor(delta_t)
        self.delta_t = delta_t
        
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
    
    def t_uniform_sample(self):
        return torch.rand(self.mc_samples) *\
            (self.target_range[1] - self.target_range[0]) + self.target_range[0]

    def forward(self, x, t):
        y = self.y_single_t_sample(x,t)

        #create uniform samples in target range
        t_random_array = self.t_uniform_sample().to(self.device)
        y_array = self.y_multiple_t_samples(x, t_random_array)
        
        return y, y_array
    
    def integration_forward(self, x, t):
        y = self.y_single_t_sample(x,t)
        y_range = self.y_multiple_t_samples(x, self.t_range)
        
        return y, y_range


class Learned_Pdf_Integration_MLP(Learned_Pdf_MC_MLP):
    
    def forward(self, x, t):
        return super().integration_forward(x, t)
    

class Binned_Gamma_Learned_Pdf_MLP(Learned_Pdf_MC_MLP):
    
    def __init__(self, vector_dim, hidden_dims, binned_gamma_params, 
                 target_range, device, 
                 mc_samples=10, delta_t=torch.tensor(0.01), 
                 mlp_activation=nn.ReLU(), dropout=0.1, torch_seed=123):
        '''
        Layer that mixes real-valed target with input vector, to generate a
        function that depends on both inputs and target via dense mlp. This 
        produces the raw output that can be turned into a probability 
        distribution of the target with a softdelta activation. It also outputs 
        MC-sampled y-values (untransformed distribution values) for the MC 
        learned-target loss function

        Parameters
        ----------
        vector_dim : int
            dimension of input vector (can be features, or latent feature space) 
        hidden_dims : list[int]
            list of hidden dimensions in mlp that mixes vector and target
        binned_gamma_params : list[tuple(float, float)]
            list of binned gamma stats, where each element has the following
            format:
                [(alpha, theta))]
            where (alpha, theta) are the fitted gamma parameters of that bin
        target_range : list[float]
            2-element list describing the range of the target
        device : torch.device
            torch device (converts output vectors to this device)
        mc_samples : int, optional
            Number of monte-carlo integration samples The default is 10.
        delta_t : float, optional
            delta step for numeric integration if using numeric integration 
            loss function, or just for outputting the entire distribution
            during inference. The default is torch.tensor(0.01).
        mlp_activation : torch.nn.modules.Module, optional
            activation of each hidden layer in the MLP. 
            The default is nn.ReLU().
        dropout : float, optional
            Dropout used for each hidden layer in MLP. The default is 0.1.
        torch_seed : int, optional
            Torch seed value. The default is 123.

        Returns
        -------
        None.
        '''
        
        super().__init__(vector_dim, hidden_dims, target_range, device,
                         mc_samples=mc_samples, delta_t=delta_t, 
                         mlp_activation=mlp_activation, dropout=dropout,
                         torch_seed=torch_seed)
        self.binned_gamma_params = binned_gamma_params
        #create gamma distributions for each binned_gamma element
        
        self.gamma_dist_bins = [
            Gamma(alpha, 1/theta) 
            for alpha, theta in self.binned_gamma_params]
        
        self.alpha = torch.tensor([a for a,th in binned_gamma_params]).\
            to(torch.float32).to(device)
        self.theta_inv = torch.tensor([1/th for a,th in binned_gamma_params]).\
            to(torch.float32).to(device)
        
    def importance_bin_samples(self, gamma_bins):
        '''
        samples from the appropriate gamma distribution for each bin value
        and places the samples in the appropriate rows
        Parameters
        ----------
        gamma_bins : torch.tensor[int]
            1D tensor value describing which target bin the sample falls under,
            of shape [batch_size]

        Returns
        -------
        target_sample: torch.tensor[float]: 2D array of gamma samples of shape 
            [batch_size, self.mc_samples]
        proposal_dist_vals: torch.tensor[float]: 2D array of the gamma pdf
            value, of shape [batch_size, self.mc_samples] which is the corresponding
            proposal distribution value for each target_sample value, to be used
            for importance sampling. Each row uses the proper gamma parameters
            corresponding to that row

        '''
        #set alpha and 1/theta to be the gamma param elements of gamma_bins   
        alpha = self.alpha[gamma_bins]
        theta_inv = self.theta_inv[gamma_bins]
        
        #get self.mc_samples samples of a Gamma distribution with the
        #appropriate distribution values for the bin of each batch sample
        res = Gamma(alpha, theta_inv).sample([self.mc_samples]).T
        
        #calculate the gamma pdf values for res values as the proposal distribution values
        alpha = alpha.unsqueeze(1)
        theta_inv = theta_inv.unsqueeze(1)
        q = theta_inv**alpha / torch.exp(torch.lgamma(alpha)) *\
            res**(alpha-1) * torch.exp(-theta_inv * res)
        
        return res, q
    
    def y_q_importance_samples(self, x, gamma_bins):
        
        #get [N, self.mc_samples] gamma samples and reshape to [N, self.mc_samples, 1]
        t_array, q_array = self.importance_bin_samples(gamma_bins)
        t_array = t_array[:,:,None]

        #reshape x from [x.shape[0], x.shape[1]] to 
        #[x.shape[0], len(t_array), x.shape[1] repeating it along the second dimension]
        x_reshape = x.unsqueeze(1).expand(x.shape[0], t_array.shape[1], x.shape[1])
        
        #add t_array_repeat as another "feature" to x along the third dimension
        x_t_array = torch.cat([x_reshape, t_array], axis=2)

        z_array = self.mlp(x_t_array)
        y_array = self.y_layer(z_array).squeeze(-1)
        
        return y_array, q_array
    
    def importance_sampling_forward(self, x, t, gamma_bins):
        y = self.y_single_t_sample(x,t)

        #create uniform samples in target range
        y_array, q_array = self.y_q_importance_samples(x, gamma_bins)
        
        return y, y_array, q_array
    
    def forward(self, x, t, gamma_bins):
        return self.importance_sampling_forward(x, t, gamma_bins)