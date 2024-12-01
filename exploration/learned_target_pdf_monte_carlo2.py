# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
this file attempts to use Monte Carlo sampling for training and numeric
integration for scoring using utility functions
"""

import pandas as pd
import numpy as np
import torch
import random
import scipy
import gc
from collections import OrderedDict

from pprint import pprint

import matplotlib.pyplot as plt

import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import functions.layers as lyr
import functions.loss_function as lf
import functions.scoring as score
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


# %% torch device
device = torch.device('cuda:0')

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
    

# %% create dataset
batch_size = 256

train_dataset = mlp_dataset(X_train, t_train)
test_dataset = mlp_dataset(X_test, t_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# %% create model
mlp_dims = [2048, 2048, 1024]
#mc_samples = 10
mc_samples = 30
#mc_samples = 1
dropout = 0.1
mlp_activation = nn.ReLU()
#mlp_activation = nn.Tanh()
seed = 123

target_range = [0,12]
delta_t = torch.tensor(0.1)

model = lyr.learned_pdf_mc_mlp(
    vector_dim=X_train.shape[1],
    hidden_dims=mlp_dims,
    target_range=target_range,
    device=device,
    mc_samples=mc_samples,
    delta_t = delta_t,
    mlp_activation=mlp_activation,
    dropout=dropout,
    torch_seed=seed
    )

model.to(device)

# %% set up optimization
learning_rate = 1e-5
num_epochs=100
weight_decay = 1e-4
batch_update = 10
seed = 124

torch.manual_seed(seed)
optimizer = torch.optim.AdamW(params=model.parameters(), 
                              lr=learning_rate, weight_decay=weight_decay)

loss_function = lf.LearnedPDFMCIntegration.apply
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
        loss = loss_function(y, torch.tensor(target_range), *y_range.T)
        
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
    
    test_epoch_loss = score.learned_pdf_eval(model, loss_function, test_dataloader, device)
    print(f'test loss after epoch {epoch+1}: {test_epoch_loss}')
    print()
    
    
# %%
p, p_dist, t, t_ev = score.learned_pdf_score(model, test_dataset)

print(r2_score(t, t_ev)) #0.7470864057540894

# %%
t_range = model.t_range.detach().cpu().numpy()
delta_t_np = delta_t.detach().cpu().numpy()

# %% plot learned pdf function
def plot_dist(p_dist, t=None, tev=None, t_range=t_range):
        
    fig, ax = plt.subplots()
    ax.plot(t_range, p_dist)
    
    ax.set(xlabel='t', ylabel='learned target dist',
           title = 'learned target distribution')
    ax.grid()
    
    if t:
        plt.axvline(x=t, color='b')
    
    if tev:
        plt.axvline(x=tev, color='r')
    
    plt.show()
    
# %% sample a result and show the plot

idx = random.randrange(len(test_df))
print(f'index: {idx}')

x = X_test[idx,:]

feature_dict = {feature: val for feature, val in zip(features, x)}
feature_dict['area'] =\
    scaler.inverse_transform(np.array([feature_dict['area_scaled']]).reshape(-1,1))[0][0]
del feature_dict['area_scaled']
pprint(feature_dict)

t_sample = t[idx]
p_dist_sample = p_dist[idx,:]
tev_sample = t_ev[idx]

print(f'target: {t_sample}')
print(f'E[t]: {tev_sample}')

plot_dist(p_dist_sample, t_sample, tev_sample, t_range)

# %% construct a sample and plot its distribution
manual_features = OrderedDict(
    bedrooms = 1, #1-3
    grade = 2, #1-5
    has_basement = 0,
    renovated = 0,
    nice_view = 0,
    perfect_condition = 0,
    real_bathrooms = 2, #1-3
    has_lavatory=0,
    single_floor=1,
    month=11,
    quartile_zone=3, #1-4
    year_from_2014 = 0, #0,1
    area_scaled = 0. #generally -2 to +3
    )

x_manual = torch.tensor(list(manual_features.values()))[None,:].to(device)
t_manual = torch.tensor([0.]).to(device)

#integration forward
with torch.no_grad():
    model.eval()
    _, y_range_manual = model.integration_forward(x_manual, t_manual)
model.train(True)

y_range_manual = y_range_manual.squeeze(0)
e_y = torch.exp(y_range_manual).cpu().detach().numpy()
p_dist_manual = e_y / sum(e_y * delta_t_np)

t_ev_manual = (e_y * delta_t_np * t_range).sum() / (e_y *delta_t_np).sum()

print(f'E[t] = {t_ev_manual}')

plot_dist(p_dist_manual, t=None, tev=t_ev_manual, t_range=t_range)
