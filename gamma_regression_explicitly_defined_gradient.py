
import pandas as pd
import numpy as np
import torch
import random
import scipy

from pprint import pprint

import matplotlib.pyplot as plt

import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from backpack import backpack, extend
from backpack.extensions import BatchGrad

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

# %% write custom gamma loss function
class GammaLoss(Function):
    
    @staticmethod
    def forward(ctx, logtheta, logk, t, *params):
        #import pdb; pdb.set_trace()
        theta = torch.clamp(torch.exp(logtheta), min=0, max=100).to(device)
        k = torch.clamp(torch.exp(logk), min=0, max=100).to(device)
        ctx.save_for_backward(t, theta, k, logtheta, logk, *params)
        loss = torch.lgamma(k) + k * torch.log(theta) -\
            (k-1)*torch.log(t) + t/theta
        return loss.mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        t, theta, k, logtheta, logk, *params = ctx.saved_tensors
        #import pdb; pdb.set_trace()
        custom_grads = []
        
        
        logtheta_grad = torch.autograd.grad(outputs=logtheta, inputs=params, 
                                       grad_outputs=torch.ones_like(logtheta), 
                                       allow_unused=True, retain_graph=True)
        #this function sums the gradients over the batch dimension (dimension of outputs)
        #we need to individually multiply each gradient with the derivative of loss WRT outputs
        
        logk_grad = torch.autograd.grad(outputs=logk, inputs=params, 
                                   grad_outputs=torch.ones_like(logk), 
                                   allow_unused=True, retain_graph=True)
        
        for param, logtheta_grad_element, logk_grad_element in zip(params, logtheta_grad, logk_grad):
            
            grad_via_logtheta = logtheta_grad_element * (
                k - t / theta
                ).sum() if logtheta_grad_element is not None else 0.
            grad_via_logk = logk_grad_element * (
                k * torch.digamma(k) + k * torch.log(theta / t)
                ).sum() if logk_grad_element is not None else 0.
            
            custom_grads.append(grad_via_logtheta + grad_via_logk)
            
        #only return gradient wrt the weights directly    
        return None, None, None, *custom_grads
    
# %% 
def gamma_loss_with_model_input(logtheta, logk, t, model):
    params = [param for param in model.parameters()]
    return GammaLoss.apply(logtheta, logk, t, *params)
#may need to define this as the forward method of a custom nn.module

#class gamma_loss_with_model(torch.nn.Module):
#    def forward(self, logtheta, logk, t, model):
#        params = [param for param in model.parameters()]
#        return GammaLoss.apply(logtheta, logk, t, *params)
    
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
class gamma_mlp(nn.Module):
    def __init__(self, vector_dim, mlp_dims, mlp_activation=nn.ReLU(), dropout=0.1):
        super().__init__()
        
        mlp_dims = [vector_dim] + mlp_dims
        mlp_layers = [
            layer
            for prev_num, num in zip(mlp_dims[0:-1], mlp_dims[1:])
            for layer in [nn.Linear(prev_num, num), nn.Dropout(dropout), mlp_activation]
            ]
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_layer = nn.Linear(mlp_dims[-1], 2)
        #self.logtheta_layer = nn.Linear(mlp_dims[-1], 1)
        #self.logk_layer = nn.Linear(mlp_dims[-1], 1)
        
    def forward(self, x):
        z = self.mlp(x)
        output = self.output_layer(z)
        return output
        logtheta = output[0]
        logk = output[1]
        #logtheta = self.logtheta_layer(z)
        #logk = self.logk_layer(z)
        #return logtheta.squeeze(-1), logk.squeeze(-1)
    
# %% function to evaluate dataset
def eval_dataset(model, loss_function, dataloader, device):

    model.train = False
    epoch_loss = 0.0
    for x, t in dataloader:
        x, t = x.to(device), t.to(device)
        logtheta, logk = model(x)
        logtheta, logk = logtheta.to(device), logk.to(device)
        with torch.no_grad():
            loss = loss_function(logtheta, logk, t, model)
        batch_loss = loss.item()
        epoch_loss += batch_loss
        
    epoch_loss /= len(dataloader)
    model.train = True
    return(epoch_loss)

# %% function to get batch-size gradient of output with respect to weights
def compute_per_sample_gradients(model, logtheta, logk):
    logtheta_per_sample_grads = {}
    logk_per_sample_grads = {}
    
    # Create a backpack context with BatchGrad to track per-sample gradients
    with backpack(BatchGrad()):
        # Perform backward pass on the output directly
        logtheta.backward(torch.ones_like(logtheta), retain_graph=True)
        
        # Collect per-sample gradients for each parameter
        for name, param in model.named_parameters():
            if param.grad_batch is not None:
                logtheta_per_sample_grads[name] = param.grad_batch.clone()
            else:
                logtheta_per_sample_grads[name] = None

    return logtheta_per_sample_grads
    
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

model = gamma_mlp(
    vector_dim=X_train.shape[1],
    mlp_dims=mlp_dims,
    mlp_activation=mlp_activation,
    dropout=dropout
    )

model.to(device)

model = extend(model)

# %% set up optimization
learning_rate = 1e-5
num_epochs=50
weight_decay = 1e-3
batch_update = 10
seed = 124

torch.manual_seed(seed)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

#loss_function = gamma_loss_with_model()
loss_function = gamma_loss_with_model_input

# %%
x,t = next(iter(train_dataloader))
x,t = x.to(device), t.to(device)

output = model(x)

logtheta, logk = output[:,0], output[:,1]

params = list(model.parameters())

#with backpack(BatchGrad()):
#    output.backward(torch.ones_like(output), retain_graph=True)

with backpack(BatchGrad()):
    logtheta.backward(torch.ones_like(logtheta))
    
logtheta_grad = [param.grad_batch for param in params]

with backpack(BatchGrad()):
    logk.backward(torch.ones_like(logk), retain_graph=True)
    
logk_grad = [param.grad_batch for param in params]
# %% perform training loop

#torch.cuda.empty_cache()
#gc.collect()

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
        logtheta, logk = model(x)
        
        #compute loss
        loss = loss_function(logtheta, logk, t, model)
        
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
def score(model, test_dataset, batch_size=256):
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    logtheta = np.array([])
    logk = np.array([])
    for x, t in dataloader:
        x,t = x.to(device), t.to(device)
        with torch.no_grad():
            model.train = False
            out_logtheta, out_logk = model(x)
        model.train = True
        out_logtheta, out_logk = out_logtheta.detach().cpu().numpy(), out_logk.detach().cpu().numpy()
        logtheta = np.concatenate([logtheta, out_logtheta])
        logk = np.concatenate([logk, out_logk])
        
    t = test_dataset[:][1].detach().cpu().numpy()
        
    return logtheta, logk, t

def score_values(model, test_dataset, rescale=True, batch_size=256):
    logtheta, logk, t = score(model, test_dataset, batch_size)
    theta = np.exp(logtheta)
    k = np.exp(logk)
    mean = k * theta
    mode = np.where(k>=1, (k-1)*theta, 0)
    std = np.sqrt(k * theta**2)
    
    if rescale:
        mean *= 1e5
        mode *= 1e5
        std *= 1e5
        t *= 1e5
    
    return pd.DataFrame({
        'theta': theta,
        'k': k,
        'target': t,
        'mean_pred': mean,
        'mode_pred': mode,
        'std_pred': std
        })

# %% score and evalute test set

scored_test = score_values(model, test_dataset, rescale=True)

print(r2_score(scored_test['target'], scored_test['mean_pred']))


# %% plot gamma function
def plot_gamma(theta, k, true_val=None, x_max=20, x_step=0.01):
    
    x = np.arange(0.0, x_max+x_step, x_step)
    y = x**(k-1) * np.exp(-x / theta) / scipy.special.gamma(k) / theta**k
    
    fig, ax = plt.subplots()
    ax.plot(x,y)
    
    ax.set(xlabel='x', ylabel=f'gamma(x; {round(k,2)}, {round(theta,2)})',
           title = f'Gamma Distribution with k={round(k,2)}, theta={round(theta,2)}')
    ax.grid()
    
    if true_val:
        plt.axvline(x=true_val, color='b')
    
    plt.show()
    
# %% sample a result and show the plot

idx = random.randrange(len(test_df))
print(f'index: {idx}')

x = X_test[idx,:]

pprint({feature: val for feature, val in zip(features, x)})

scored_sample = scored_test.iloc[idx,:]
pprint(scored_sample)

true_val = scored_sample['target'] / 1e5

plot_gamma(scored_sample['theta'], scored_sample['k'], true_val=true_val)
