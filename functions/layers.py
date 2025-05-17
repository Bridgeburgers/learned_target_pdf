# -*- coding: utf-8 -*-
"""
Custom torch layers for learned-pdf 
"""

import torch
import torch.nn as nn
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
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
    
    
class Gamma_Gaussian_Mixture_EBM_Sampler(Learned_Pdf_MC_MLP):
    
    def __init__(self, vector_dim, hidden_dims, gamma_params,
                 target_range, device, 
                 normal_prob=0.2, normal_sigma_divider=4.,
                 uniform_prob=0.2, uniform_range=1000.,
                 mc_samples=10, delta_t=torch.tensor(0.01), 
                 mlp_activation=nn.ReLU(), dropout=0.1, torch_seed=123):
        '''
        Energy-based neural network layer that generates importance sampling
        samples from a mixture of gamma distribution fit on the whole data, 
        a Gaussian centered at a given data point, and a uniform distribution

        Parameters
        ----------
        vector_dim : int
            dimension of input vector (can be features, or latent feature space) 
        hidden_dims : list[int]
            list of hidden dimensions in mlp that mixes vector and target
        gamma_params: tuple(float, float) whose first element is alpha
            and second is theta, for the gamma distribution fit on the training
            set
        target_range : list[float]
            2-element list describing the range of the target
        device : torch.device
            torch device (converts output vectors to this device)
        normal_prob: float
            probability of normal distribution in the gamma-normal mixture 
            distribution for importance sampling
        normal_sigma_divider: float
            the sampled normal distribution for importance sampling will have
            parameters mu=x, sigma=x/(normal_sigma_divider), where x is the
            value of the target sample
        uniform_prob: float
            probability of uniform distribution in the mixture distribution
        uniform_range: float
            end point range of sampled uniform distribution (starting range
            is always 0)
        mc_samples : int, optional
            Number of importance samples. The default is 10.
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
        self.gamma_params = gamma_params
        #create gamma distributions for each binned_gamma element
        
        self.normal_prob = normal_prob
        self.normal_sigma_divider = normal_sigma_divider
        
        self.uniform_prob = uniform_prob
        self.uniform_range = uniform_range
        
        self.alpha = torch.tensor(gamma_params[0]).\
            to(torch.float32).to(device)
        self.theta_inv = torch.tensor(1 / self.gamma_params[1]).\
            to(torch.float32).to(device)
        
    def importance_gamma_samples(self, batch_size: int):
        '''
        generates (batch_num * mc_samples) tensor of gamma samples, and their
        respective density value to be used for importance sampling

        Parameters
        ----------
        batch_size: int
            number of rows for which to generate mc_samples gamma samples

        Returns
        -------
        target_sample: torch.tensor[float]: 2D array of gamma samples of shape 
            [batch_size, self.mc_samples]
        '''
        
        #get batch_size x self.mc_samples samples of a Gamma distribution
        res = Gamma(self.alpha, self.theta_inv).\
            sample([batch_size, self.mc_samples])
        
        return res
    
    def gamma_density(self, res: torch.tensor):
        '''
        calculates corresponding probability density for a set of gamma samples
        '''
        q = self.theta_inv**self.alpha / torch.exp(torch.lgamma(self.alpha)) *\
            res**(self.alpha-1) * torch.exp(-self.theta_inv * res)
            
        return torch.nan_to_num(q, 0.0) #this is for when normal draw gives sample < 0
    
    
    def importance_norm_samples(self, t):
        '''
        generates (batch_num * mc_samples) tensor of Normal samples and their
        respective density values to be used for importance sampling, where
        each row is sampled from a Normal distribution whose mean corresponds
        to that element of t, and SD is t/self.normal_sigma_divider

        Parameters
        ----------
        t : torch.tensor[float]
            1D tensor of target values to inform Normal distribution

        Returns
        -------
        target_sample: torch.tensor[float]: 2D array of Normal samples of shape 
            [batch_size, self.mc_samples]
        '''
        #define Normal parameters
        mu = t
        sigma = t/self.normal_sigma_divider
        
        #get Normal samples where each row corresponds to values from x
        res = Normal(mu, sigma).sample([self.mc_samples]).T
        return res
    
    def normal_density(self, res: torch.tensor, t: torch.tensor):
        '''
        calculates corresponding probability density for a set of normal samples
        '''
        #define Normal parameters
        mu = t
        sigma = t/self.normal_sigma_divider
        
        #calculate the normal pdf values
        mu_expanded = mu.unsqueeze(1).expand(mu.shape[0], self.mc_samples)
        sigma_expanded = sigma.unsqueeze(1).expand(sigma.shape[0], self.mc_samples)
        q = torch.exp(-(mu_expanded - res)**2 / 2 / sigma_expanded**2) /\
            torch.sqrt(2 * torch.pi * sigma_expanded**2)
            
        return q
    
    def importance_uniform_samples(self, batch_size: int):
        '''
        generates (batch_num * mc_samples) tensor of uniform samples

        Parameters
        ----------
        batch_size: int
            number of rows for which to generate mc_samples gamma samples

        Returns
        -------
        target_sample: torch.tensor[float]: 2D array of gamma samples of shape 
            [batch_size, self.mc_samples]
        proposal_dist_vals: torch.tensor[float]: 2D array of the gamma pdf
            value, of shape [batch_size, self.mc_samples] which is the corresponding
            proposal distribution value for each target_sample value, to be used
            for importance sampling.

        '''
        
        #get batch_size x self.mc_samples samples of a uniform distribution
        res = torch.rand([batch_size, self.mc_samples]) * self.uniform_range
        return res.to(self.device)
    
    def uniform_density(self, res: torch.tensor):
        '''
        calculates corresponding probability density for a set of uniform samples
        '''
        return (torch.ones(res.shape) / self.uniform_range).to(self.device)
    
    def importance_mixture_samples(self, t):
        '''
        generates importance samples for a mixture of Gaussian and gamma
        distribution where the Gaussian values are centered around t, and
        their respective density values

        Parameters
        ----------
        t : torch.tensor[float]
            1D tensor of target values to inform Normal distribution

        Returns
        -------
        target_sample: torch.tensor[float]: 2D array of Normal-gamma mixture 
            samples of shape [batch_size, self.mc_samples]
        proposal_dist_vals: torch.tensor[float]: 2D array of the Normal-gamma 
            mixture pdf value, of shape [batch_size, self.mc_samples] which is 
            the corresponding proposal distribution value for each 
            target_sample value, to be used for importance sampling.

        '''
        gamma_res = self.importance_gamma_samples(len(t))
        normal_res = self.importance_norm_samples(t)
        uniform_res = self.importance_uniform_samples(len(t))
        
        #generate a mixture distribution output of the Normal and Gamma draws
        normal_sample_tensor = Bernoulli(self.normal_prob).\
            sample(normal_res.shape).to(self.device)
        uniform_sample_tensor = Bernoulli(self.uniform_prob / (1 - self.normal_prob)).\
            sample(uniform_res.shape).to(self.device)
            
        res = normal_sample_tensor * normal_res +\
            (1-normal_sample_tensor) * uniform_sample_tensor * uniform_res +\
            (1-normal_sample_tensor) * (1-uniform_sample_tensor) * gamma_res
        
        #calculate gamma and normal probability densities
        gamma_q = self.gamma_density(res)
        normal_q = self.normal_density(res, t)
        uniform_q = self.uniform_density(res)
        
        #calculate q
        q = self.normal_prob * normal_q +\
            self.uniform_prob * uniform_q +\
            (1 - self.normal_prob - self.uniform_prob) * gamma_q
        
        return res, q
        
    
    def y_q_importance_samples(self, x, t):
        
        #get [N, self.mc_samples] gamma samples and reshape to [N, self.mc_samples, 1]
        t_array, q_array = self.importance_mixture_samples(t)
        t_array = t_array[:,:,None]

        #reshape x from [x.shape[0], x.shape[1]] to 
        #[x.shape[0], mc_samples, x.shape[1] repeating it along the second dimension]
        x_reshape = x.unsqueeze(1).expand(x.shape[0], t_array.shape[1], x.shape[1])
        
        #add t_array_repeat as another "feature" to x along the third dimension
        x_t_array = torch.cat([x_reshape, t_array], axis=2)

        z_array = self.mlp(x_t_array)
        y_array = self.y_layer(z_array).squeeze(-1)
        
        return y_array, q_array
    
    def importance_sampling_forward(self, x, t):
        y = self.y_single_t_sample(x,t)

        #create uniform samples in target range
        y_array, q_array = self.y_q_importance_samples(x, t)
        
        return y, y_array, q_array
    
    def forward(self, x, t):
        return self.importance_sampling_forward(x, t)
    
    
class Gaussian_EBM_Sampler(Learned_Pdf_MC_MLP):
    
    def __init__(self, vector_dim, hidden_dims,
                 target_range, device, mu, sigma,
                 mc_samples=10, delta_t=torch.tensor(0.01), 
                 mlp_activation=nn.ReLU(), dropout=0.1, torch_seed=123):
        '''
        Energy-based neural network layer that generates importance sampling
        samples from a Gaussian proposal distribution

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
        mu: float
            mean of the proposal distribution
        sigma: float
            sd of the proposal distribution
        mc_samples : int, optional
            Number of importance samples. The default is 10.
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
        self.mu = torch.tensor(mu).to(torch.float32).to(device)
        self.sigma = torch.tensor(sigma).to(torch.float32).to(device)
    
    
    def importance_norm_samples(self, batch_size):
        '''
        generates (batch_num * mc_samples) tensor of Normal samples and their
        respective density values to be used for importance sampling
    

        Returns
        -------
        target_sample: torch.tensor[float]: 2D array of Normal samples of shape 
            [batch_size, self.mc_samples]
        density: torch.tensor[float]: 2D array of density values corresponding
            to the samples from target_sample
        '''

        
        #get Normal samples where each row corresponds to values from x
        res = Normal(self.mu, self.sigma).sample([batch_size, self.mc_samples])
        q = torch.exp(-(self.mu - res)**2 / 2 / self.sigma**2) /\
            torch.sqrt(2 * torch.pi * self.sigma**2)
        return res, q       
    
    def y_q_importance_samples(self, x, t):
        
        #get [N, self.mc_samples] normal samples and reshape to [N, self.mc_samples, 1]
        t_array, q_array = self.importance_norm_samples(len(t))
        t_array = t_array[:,:,None]

        #reshape x from [x.shape[0], x.shape[1]] to 
        #[x.shape[0], mc_samples, x.shape[1] repeating it along the second dimension]
        x_reshape = x.unsqueeze(1).expand(x.shape[0], t_array.shape[1], x.shape[1])
        
        #add t_array_repeat as another "feature" to x along the third dimension
        x_t_array = torch.cat([x_reshape, t_array], axis=2)

        z_array = self.mlp(x_t_array)
        y_array = self.y_layer(z_array).squeeze(-1)
        
        return y_array, q_array
    
    def importance_sampling_forward(self, x, t):
        y = self.y_single_t_sample(x,t)

        #create uniform samples in target range
        y_array, q_array = self.y_q_importance_samples(x, t)
        
        return y, y_array, q_array
    
    def forward(self, x, t):
        return self.importance_sampling_forward(x, t)