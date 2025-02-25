import numpy as np
import matplotlib.pyplot as plt

import torch
import transformers

import functions.data_simulator as ds

import functions.layers as lyr
import functions.loss_function as lf
import functions.utility as utl

# %% set synthetic distribution parameters
# mu1 = a1 sin(b11 x1 + b12 x2 + phi1)
#sigma1 = d1 + c1 sqrt(x1^2 + x2^2) / (abs(x5) + 0.05)
#mu2 = a2 sin(b21 x3 + b22 x4 + phi2)
#sigma2 = d2 + c2 sqrt(x3^2 + x4^2) / (abs(x5) + 0.05)
#pi = 1 / (1 + exp(-a3 x5 + b3))

params = dict(
    a1 = 8.,
    b1 = 10.,
    phi1 = 0,
    c1 = 2.,
    d1 = 1.,
    a2 = 4.,
    b2 = 5.,
    phi2 = np.pi/2,
    c2 = 2.,
    d2 = 1.,
    a3 = -1.,
    b3 = 0.2,
    delta_t = 0.01,
    target_range = [-20., 20.],
)

n_samples = 100000

# %% create data simulator
sim = ds.Bimodal_Gaussian_Target(**params)
t_range = sim.t_range

# %% create simulated data
np.random.seed(100)
X, t = sim.generate_data(n_samples)

# %% create train-test split
train_split = 0.8
train_inds = np.random.choice(len(X), round(len(X)*train_split), replace=False)
test_inds = np.array(list(set(range(len(X))) - set(train_inds)))

X_train = X[train_inds,:]
X_test = X[test_inds,:]
t_train = t[train_inds]
t_test = t[test_inds]

# %% specify the name of the huggingface model to load
model_name = "distilbert-base-uncased"
#model_name = 'bert-base-uncased'
#model_name = 'roberta-base'

# %% torch device
device = torch.device('cuda:2')


# %% load BERT
bert = transformers.AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

bert = bert.to(device)

#print number of parameters
n_params = sum([x.numel() for x in list(bert.parameters())])
print(n_params)