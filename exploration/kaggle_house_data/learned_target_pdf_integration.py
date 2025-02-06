# -*- coding: utf-8 -*-
"""
this file attempts to use numeric integration during training and scoring for
learned target pdf
"""

import pandas as pd
import numpy as np
import torch
import random
import scipy
import gc

from pprint import pprint

import matplotlib.pyplot as plt

import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
# %% load data
dir = 'D:/Documents/Data/kaggle_house_data/'
train_file = 'df_train.csv'
test_file = 'df_test.csv'

train_df = pd.read_csv(dir + train_file)
test_df = pd.read_csv(dir + test_file)

# %% create year
for df in train_df, test_df:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['year_from_2014'] = df['year'] - 2014
    
# %% scale area column
scaler = StandardScaler()
train_df['area_scaled'] = scaler.fit_transform(train_df[['living_in_m2']])
test_df['area_scaled'] = scaler.transform(test_df[['living_in_m2']])

# %% convert bool vars to binary
bool_vars = (train_df.columns[train_df.dtypes==bool]).values
for df in train_df, test_df:
    df[bool_vars] = df[bool_vars].astype(int)
    
# %% rescale target in units of $10^5
for df in train_df, test_df:
    df['price'] /= 1e5

# %% create arrays for training/testing
features = ['bedrooms', 'grade', 'has_basement', 'renovated', 'nice_view',
            'perfect_condition', 'real_bathrooms', 'has_lavatory', 
            'single_floor', 'month', 'quartile_zone', 'year_from_2014',
            'area_scaled']
target = 'price'

X_train = train_df[features].values
t_train = train_df[target].values

X_test = test_df[features].values
t_test = test_df[target].values

# %%

target_range = [0,12]
delta_t = 0.2
t_range = np.arange(target_range[0], target_range[1]+delta_t, delta_t)

# %%
#def numeric_integral(function_tensor, delta_t):
#    return torch.sum(function_tensor[1:] + function_tensor[0:-1],axis=1)/2 * delta_t

def numeric_integral(function_tensor, delta_t):
    return torch.sum(function_tensor * delta_t, axis=1)

# %% torch device
device = torch.device('cuda:0')

# %% write custom loss for learned target PDF using numeric integration
class LearnedPDFIntegration(Function):
    
    @staticmethod
    def forward(ctx, y, *y_range):
        #import pdb; pdb.set_trace()
        y_range = torch.stack(y_range, dim=1).to(device)
        Z = numeric_integral(torch.exp(y_range), delta_t).to(device)
        ctx.save_for_backward(y, y_range, Z)
        loss = -y + torch.log(Z)
        return loss.mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        #import pdb; pdb.set_trace()
        y, y_range, Z = ctx.saved_tensors
        
        y_grad = -1 * grad_output * torch.ones(y.shape).to(device)
        y_range_grad = torch.div(torch.exp(y_range) * delta_t, Z.reshape(-1,1))
        
        return y_grad, *y_range_grad.T

    
    
# %% create nn dataset class
class mlp_dataset(Dataset):
    def __init__(self, X, t):
        assert X.shape[0] == len(t)
        self.X = X
        self.t = t
        
    def __len__(self):
        return len(self.t)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx,:]).float(), torch.tensor(self.t[idx]).float()
    
# %% create mlp class
#MAKE THIS TAKE x AND t AS INPUT
class learned_pdf_integral_mlp(nn.Module):
    def __init__(self, vector_dim, mlp_dims, target_range, delta_t=0.01, 
                 mlp_activation=nn.ReLU(), dropout=0.1):
        super().__init__()
        
        self.t_range = torch.arange(target_range[0], target_range[1]+delta_t, delta_t).to(device)
        self.n_range = len(self.t_range)
        
        mlp_dims = [vector_dim + 1] + mlp_dims
        mlp_layers = [
            layer
            for prev_num, num in zip(mlp_dims[0:-1], mlp_dims[1:])
            for layer in [nn.Linear(prev_num, num), nn.Dropout(dropout), mlp_activation]
            ]
        self.mlp = nn.Sequential(*mlp_layers)
        self.y_layer = nn.Linear(mlp_dims[-1], 1)
        #self.y_range_layer = nn.Linear(mlp_dims[-1], self.n_range)
        
#    def forward(self, x, t):
#        x_t = torch.cat([x, t.reshape(-1,1)], axis=1)
#        z = self.mlp(x_t)
#        y = self.y_layer(z)
#        y_range = self.y_range_layer(z)
#        return y.squeeze(-1), y_range

    def forward(self, x, t):
        x_t = torch.cat([x, t.reshape(-1,1)], axis=1)
        z = self.mlp(x_t)
        y = self.y_layer(z).squeeze(-1)
        #t_range_repeat = self.t_range.repeat(len(t), 1)[:,:,None]
        t_range_repeat = self.t_range.expand(len(t), len(t_range))[:,:,None]
        x_reshape = x.unsqueeze(1).expand(x.shape[0], t_range_repeat.shape[1], x.shape[1])
        
        x_t_range = torch.cat([x_reshape, t_range_repeat], axis=2)
        z_range = self.mlp(x_t_range)
        y_range = self.y_layer(z_range).squeeze(-1)
        
        return y, y_range
    
        #creating x_reshape and/or t_range (or x_t_range) takes forever,
        #find a more efficient way to do this
# %% 
layer = nn.Linear(3,1)
x = torch.randn([10,2])
t = torch.randn([10,8])
t_reshape = t[:,:, None]
#x_reshape = x.unsqueeze(2).expand(10,2,8)
x_reshape = x.unsqueeze(1).expand(10,8,2)

x_t = torch.cat([x_reshape, t_reshape], axis=2)
i=0
layer(x_t[:,:,i])
layer(torch.cat([x, t[:,i].reshape(-1,1)], axis=1))

layer(x_t)


x_t_reshape = x_t.reshape(-1, 3)
y_range = layer()
# %% function to evaluate dataset
def eval_dataset(model, loss_function, dataloader, device):
    
    model.eval()
    epoch_loss = 0.0
    for x, t in dataloader:
        x, t = x.to(device), t.to(device)
        y, y_range = model(x,t)
        y, y_range = y.to(device), y_range.to(device)
        loss = loss_function(y, *y_range.T)
        batch_loss = loss.item()
        epoch_loss += batch_loss
        
    epoch_loss /= len(dataloader)
    model.train(True)
    return(epoch_loss)
    
# %% create dataset
batch_size = 256

train_dataset = mlp_dataset(X_train, t_train)
test_dataset = mlp_dataset(X_test, t_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# %% create model
mlp_dims = [2048, 2048, 1024]
dropout = 0.1
#mlp_activation = nn.ReLU()
mlp_activation = nn.Tanh()

model = learned_pdf_integral_mlp(
    vector_dim=X_train.shape[1],
    mlp_dims=mlp_dims,
    target_range=target_range,
    delta_t = delta_t,
    mlp_activation=mlp_activation,
    dropout=dropout
    )

model.to(device)

# %% set up optimization
learning_rate = 1e-5
num_epochs=50
weight_decay = 1e-3
batch_update = 10
seed = 124

torch.manual_seed(seed)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

loss_function = LearnedPDFIntegration.apply

# %% perform training loop

torch.cuda.empty_cache()
gc.collect()

for epoch in range(num_epochs):
    print(f'Starting epoch {epoch+1}')
    
    #set current loss
    current_loss = 0.0
    batch_count = 0
    epoch_loss = 0.0
    
    #iterate over dataloader for training data
    for i, data in enumerate(train_dataloader,0):
        
        #print(i)
        
        #get inputs
        x, t = data
        x, t = x.to(device), t.to(device)
        
        #zero the gradients
        optimizer.zero_grad()
        
        #get model outputs
        y, y_range = model(x, t)
        y, y_range = y.to(device), y_range.to(device)
        
        #compute loss
        loss = loss_function(y, *y_range.T)
        
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
    
    test_epoch_loss = eval_dataset(model, loss_function, test_dataloader, device)
    print(f'test loss after epoch {epoch+1}: {test_epoch_loss}')
    print()
    
    
# %% scoring function
def score(model, test_dataset, batch_size=1024):
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    y = np.array([])
    y_range = None
    for x, t in dataloader:
        x,t = x.to(device), t.to(device)
        with torch.no_grad():
            model.train = False
            out_y, out_yrange = model(x,t)
        model.train = True
        out_y, out_yrange = out_y.detach().cpu().numpy(), out_yrange.detach().cpu().numpy()
        y = np.concatenate([y, out_y])
        y_range = np.concatenate([y_range, out_yrange], axis=0) if y_range is not None else out_yrange
        
    t = test_dataset[:][1].detach().cpu().numpy()
        
    return y, y_range, t

def score_values(model, test_dataset, batch_size=1024):
    y, y_range, t = score(model, test_dataset, batch_size)
    e_yrange = np.exp(y_range)
    Z = (e_yrange * delta_t).sum(axis=1)
    p_dist = e_yrange/Z.reshape(-1,1)
    p = np.exp(y) / Z
    t_ev = (e_yrange * delta_t * t_range).sum(axis=1) / Z
    
    return p, p_dist, t, t_ev

# %%
p, p_dist, t, t_ev = score_values(model, test_dataset)

print(r2_score(t, t_ev)) #0.7498564316198878

# %% plot learned pdf function
def plot_dist(p_dist, t, tev=None, t_range=t_range):
        
    fig, ax = plt.subplots()
    ax.plot(t_range, p_dist)
    
    ax.set(xlabel='t', ylabel='learned target dist',
           title = 'learned target distribution')
    ax.grid()
    
    plt.axvline(x=t, color='b')
    
    if tev:
        plt.axvline(x=tev, color='r')
    
    plt.show()
    
# %% sample a result and show the plot

idx = random.randrange(len(test_df))
print(f'index: {idx}')

x = X_test[idx,:]

pprint({feature: val for feature, val in zip(features, x)})

t_sample = t[idx]
p_dist_sample = p_dist[idx,:]
tev_sample = t_ev[idx]

print(f'target: {t_sample}')
print(f'E[t]: {tev_sample}')

plot_dist(p_dist_sample, t_sample, tev_sample, t_range)
