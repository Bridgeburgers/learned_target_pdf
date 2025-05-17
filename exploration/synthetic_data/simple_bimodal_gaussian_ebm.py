# -*- coding: utf-8 -*-
"""
Train an EBM on simulated data that has bimodal distributions for some samples
but the overall distribution is unimodal
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import time
import gc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import functions.data_simulator as ds

import functions.layers as lyr
import functions.loss_function as lf
import functions.utility as utl

# %% set data simulator parameters
# =============================================================================
# mu1 = self.a * x1
# mu2 = self.b * x2
# 
# sigma1 = 1. + self.c * np.abs(x3)
# sigma2 = 1. + self.d * np.abs(x4)
# 
# pi = 1 / (1 + np.exp(-self.e * x5))
# 
# =============================================================================
params = dict(
    a=8.,
    b=8.,
    c=1.,
    d=1.,
    e=3.,
    delta_t = 0.01,
    target_range = [-20., 20.]
)

n_samples = 10000000

np.random.seed(500)
# %% create data simulator
sim = ds.Simple_Bimodal_Gaussian_Target(**params)

# %%
t_range = sim.t_range
# %% create simulated data
X, t = sim.generate_data(n_samples)

# %% generate random index, plot the target distribution, target, and distribution ev

idx = np.random.choice(len(X), 1)[0]

xi = X[idx]
ti = t[idx]

dist = sim.target_dist(*xi)
tev = sim.target_ev(*xi)

mu1, mu2, sigma1, sigma2, pi = sim.target_params(*xi)

fig, ax = plt.subplots()
ax.plot(t_range, dist, label='target_dist')

ax.set(xlabel='t', ylabel='true_distribution')
ax.grid()

plt.axvline(x=ti, color='b', label='target')
plt.axvline(x=tev, color='r', label='distribution_ev')
plt.legend()

plt.show()

print({
    'mu1': float(mu1),
    'sigma1': float(sigma1),
    'mu2': float(mu2),
    'sigma2': float(sigma2),
    'pi': float(pi)
})

# %% plot the overall histogram of t
plt.hist(t, bins=1000)
plt.xlim(-30, 30)
plt.show()
plt.close()

# %% split into train and test
inds = np.random.choice(n_samples, size=int(0.8*n_samples), replace=False)
test_inds = np.array(list(set(range(n_samples)) - set(inds)))

X_train = X[inds,:]
t_train = t[inds]

X_test = X[test_inds, :]
t_test = t[test_inds]

# %% torch device
device = torch.device('cuda:0')

# %% create dataset class
class bimodal_dataset(Dataset):
    def __init__(self, X, t, device):
        
        self.X = torch.tensor(X).to(torch.float32).to(device)
        self.t = torch.tensor(t).to(torch.float32).to(device)
        
    def __len__(self):
        return len(self.t)
    
    def __getitem__(self, idx):
        return self.X[idx,:], self.t[idx]
    
# %% create datasets and dataloaders
batch_size=256

train_dataset = bimodal_dataset(X_train, t_train, device)
test_dataset = bimodal_dataset(X_test, t_test, device)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# %% get the mean and sd of target distribution
mu = t.mean()
sigma = t.std()

# %% define eval function
def ebm_importance_eval(model, loss_function, dataloader, device):
        
    model.eval()
    epoch_loss = 0.0
    for data in dataloader:
        x,t = data
        with torch.no_grad():
            y, y_array, q_array =\
                model(x,t)
        y, y_array, q_array = y.to(device), y_array.to(device), q_array.to(device)
        loss = loss_function(y, q_array, *y_array.T)
        batch_loss = loss.item()
        epoch_loss += batch_loss
        
    epoch_loss /= len(dataloader)
    model.train(True)
    return(epoch_loss)

# %% set parameters for EBM layer and create EBM layer
vector_dim = 5
hidden_dims = [512, 512]
target_range = sim.target_range
mc_samples = 150
delta_t = 0.1
mlp_activation = nn.ReLU()
dropout=0.1
torch_seed=123

model = lyr.Gaussian_EBM_Sampler(
    vector_dim=vector_dim,
    hidden_dims=hidden_dims,
    target_range=target_range,
    device=device,
    mu=mu,
    sigma=2*sigma, #make the importance sample distribution twice the width of the data
    mc_samples=mc_samples,
    delta_t=delta_t,
    mlp_activation=mlp_activation,
    dropout=dropout,
    torch_seed=torch_seed
)

model = model.to(device)

model_n_params = sum([x.numel() for x in list(model.parameters())])
print(model_n_params) #67841

# %% set up training parameters
learning_rate = 1e-5
num_epochs=3
weight_decay = 1e-4
batch_update = 50
seed = 124

torch.manual_seed(seed)
optimizer = torch.optim.AdamW(params=model.parameters(), 
                              lr=learning_rate, weight_decay=weight_decay)

loss_function = lf.LearnedPDFImportanceSampling.apply
# %% perform training loop 

torch.cuda.empty_cache()
gc.collect()

train_loss = []
test_loss = []

T = time.time()

for epoch in range(num_epochs):
    print(f'Starting epoch {epoch+1}')
    
    #set current loss
    current_loss = 0.0
    batch_count = 0
    epoch_loss = 0.0
    
    #iterate over dataloader for training data
    for i, data in enumerate(train_dataloader,0):
                
        #zero the gradients
        optimizer.zero_grad()
        
        x, t = data
        
        #get model outputs
        y, y_array, q_array =\
            model(x, t)
        y, y_array, q_array = y.to(device), y_array.to(device), q_array.to(device)
        
        #compute loss
        loss = loss_function(y, q_array, *y_array.T)
        
        #do backward pass
        loss.backward()
        
        #perform optimization
        optimizer.step()
        
        #print statistics
        current_loss += loss.item()
        epoch_loss += loss.item()
        batch_count += 1
        
        if i % batch_update == batch_update-1:
            print(f'Loss after batch {i+1}: {current_loss / batch_count}')
            current_loss = 0.0
            batch_count = 0
            
    epoch_loss /= len(train_dataloader)
    print('-' * 20)
    print(f'Loss after epoch {epoch+1}: {epoch_loss}')
    
    test_epoch_loss = ebm_importance_eval(model, loss_function, test_dataloader, device)
    print(f'test loss after epoch {epoch+1}: {test_epoch_loss}')
    print()
    
    train_loss.append(epoch_loss)
    test_loss.append(test_epoch_loss)
    
T = time.time() - T
print(f'training time after {epoch+1} epochs: {T} seconds')

# training time after 5 epochs: 158.48146390914917 seconds
# %% scoring functions
def learned_pdf_predict(model, test_dataset, batch_size=1024):
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    y = np.array([])
    y_range = None
    t = np.array([])
    #device = model.device
    for i, data in enumerate(dataloader,0):
        x_i, t_i = data
        with torch.no_grad():
            model.eval()
            out_y, out_yrange = model.integration_forward(x_i,t_i)
        model.train(True)
        out_y, out_yrange = out_y.detach().cpu().numpy(), out_yrange.detach().cpu().numpy()
        y = np.concatenate([y, out_y])
        y_range = np.concatenate([y_range, out_yrange], axis=0) if y_range is not None else out_yrange
        t = np.concatenate([t, t_i.detach().cpu().numpy()])
        
    return y, y_range, t

def learned_pdf_score(model, test_dataset, batch_size=1024,
                      t_range=None, delta_t=None):
    
    if t_range is None:
        try:
            t_range = model.t_range
        except AttributeError:
            raise ValueError('t_range must be provided, or an attribute of model')
            
    if delta_t is None:
        try:
            delta_t = model.delta_t
        except AttributeError:
            raise ValueError('delta_t must be provided, or an attribute of model')
            
    if torch.is_tensor(delta_t):
        delta_t = delta_t.detach().cpu().numpy()
    if torch.is_tensor(t_range):
        t_range = t_range.detach().cpu().numpy()
    
    y, y_range, t = learned_pdf_predict(model, test_dataset, batch_size)
    e_yrange = np.exp(y_range)
    Z = (e_yrange * delta_t).sum(axis=1)
    p_dist = e_yrange/Z.reshape(-1,1)
    p = np.exp(y) / Z
    t_ev = (e_yrange * delta_t * t_range).sum(axis=1) / Z
    
    return p, p_dist, t, t_ev

# %% predict distribution for all test samples
p, p_dist, t, t_ev = learned_pdf_score(model, test_dataset)

print(r2_score(t, t_ev)) #0.6987039975212788


# %% get the mode
t_range=model.t_range.detach().cpu().numpy()
t_mode = t_range[np.argmax(p_dist, axis=1)]

print(r2_score(t, t_mode))  #0.6063393145661689

# %% get the true ev, and evaluate predicted EV wrt to it
true_ev = sim.target_ev_array(X_test, chunks=5000)

print(r2_score(true_ev, t_ev)) #0.959303261834891

# %% plot learned pdf function
def plot_dist(p_dist, t, x, tev, t_range=model.t_range.detach().cpu().numpy()):
        
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(t_range, p_dist, label='predicted_dist', color='r')
    
    ax.set(xlabel='t', ylabel='density',
           title = 'ebm distribution prediction vs true distribution',
           xlim = sim.target_range)
    ax.grid()
    
    true_dist = sim.target_dist(*x)
    true_ev = sim.target_ev(*x)
    
    ax.plot(sim.t_range, true_dist, label='true_distribution')
    #ax.set(xlabel='t', ylabel='true_distribution')

    
    plt.axvline(x=t, color='g', label=f'target: {t}')
    
    plt.axvline(x=tev, color='r', label=f'predicted ev: {tev}')
    
    plt.axvline(x=true_ev, color='b', label=f'true ev: {true_ev}')
    
    plt.legend()
    
    plt.show()
    
# %% sample a result and show the plot

idx = random.randrange(len(p))

#idx = np.random.choice(np.where(t>=400)[0],1)[0]

print(f'index: {idx}')

xi, ti = test_dataset[idx]
xi, ti = xi.cpu().detach().numpy(), ti.cpu().detach().numpy()
print(f'x: {xi}')
print(f't: {ti}')


plot_dist(p_dist[idx], t=ti, x=xi, tev=t_ev[idx])

