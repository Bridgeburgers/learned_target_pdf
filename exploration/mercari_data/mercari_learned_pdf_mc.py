
"""
training a BERT model on the mercari price suggestion data using learned target pdf
"""

import os
import gc
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import random
from pprint import pprint

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import transformers

import functions.layers as lyr
import functions.loss_function as lf
import functions.utility as utl

# %% rmsle
rmsle = lambda y_true, y_pred: np.sqrt(((np.log(y_pred + 1) - np.log(y_true + 1))**2).sum()/len(y_pred))

# %% You can specify any Hugging Face pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = "distilbert-base-uncased"
#model_name = 'bert-base-uncased'
#model_name = 'roberta-base'

# %% torch device
device = torch.device('cuda:0')


# %% load BERT
bert = transformers.AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

bert = bert.to(device)

#print number of parameters
n_params = sum([x.numel() for x in list(bert.parameters())])
print(n_params)

# %% load the Mercari data from a local path
# the data can be downloaded from https://www.kaggle.com/c/mercari-price-suggestion-challenge/data
path = 'D:/Documents/Data/kaggle_mercari_price_data/'
train_df = pd.read_csv(os.path.join(path, 'train.tsv'), sep='\t')
test_df = pd.read_csv(os.path.join(path, 'test.tsv'), sep='\t')
test_stg2_df = pd.read_csv(os.path.join(path, 'test_stg2.tsv'), sep='\t')

sample_submission_df = pd.read_csv(os.path.join(path, 'sample_submission_stg2.csv'))

# %% split train into train and eval
train_frac = 0.8
seed = 123
train, val = train_test_split(train_df, train_size=round(train_frac * len(train_df)),
                              random_state=seed, shuffle=True)

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)


# %% fill nulls for all fields that contain them

for df in train, val, test_df, test_stg2_df:
    df.loc[df['item_description'].isnull(), 'item_description'] = ''
    
# %% remove zeroes in target
for df in train, val:
    df.loc[df['price']<=0., 'price'] = 0.01
    
# %% lowercase text fields
for col in ['item_description', 'name']:
    train[col] = train[col].str.lower()
    val[col] = val[col].str.lower()
    test_df[col] = test_df[col].str.lower()
    test_stg2_df[col] = test_stg2_df[col].str.lower()
    
# %% other imputation for category and brand
label_cols = ['category_name', 'brand_name']

other_percentile = 0.98

other_imp = utl.Other_Imputer(label_cols, other_percentile, other_val='OTHER')
train = other_imp.fit_transform(train)
val = other_imp.transform(val)
test_df = other_imp.transform(test_df)
test_stg2_df = other_imp.transform(test_stg2_df)


# %% label encoding for category and brand
enc = utl.Other_Ordinal_Encoder(label_cols, 'OTHER')

train = enc.fit_transform(train)
val = enc.transform(val)
test_df = enc.transform(test_df)
test_stg2_df = enc.transform(test_stg2_df)

# %% get num unique values for category_name and brand_name
n_unique_category = train['category_name'].nunique() 
n_unique_brand = train['brand_name'].nunique() 

print(n_unique_category)
#500
print(n_unique_brand)
#815
# %% one-hot encode item_condition_id
def custom_combiner(feature, category):
    return str(feature) + "_" + str(category)

ohe = OneHotEncoder(feature_name_combiner=custom_combiner).fit(train[['item_condition_id']])

def concat_ohe_cols(df):
    Z = ohe.transform(df[['item_condition_id']]).toarray()
    Z = pd.DataFrame(Z, columns=ohe.get_feature_names_out())
    df = pd.concat([df, Z], axis=1)
    del df['item_condition_id_5']
    return df

train = concat_ohe_cols(train)
val = concat_ohe_cols(val)
test_df = concat_ohe_cols(test_df)
test_stg2_df = concat_ohe_cols(test_stg2_df)

# %% define numeric features
numeric_features = list(ohe.get_feature_names_out()[0:4]) + ['shipping']

# %% create gammma_bins (decile bins) for the target
bin_creator = utl.Gamma_Bin_Creator(num_bins=5, fit_all_higher_bins=True, quantile=True)
train['gamma_bins'] = bin_creator.fit_transform(train['price'].values)
val['gamma_bins'] = bin_creator.transform(val['price'].values)
# %% create dataset class
class Mercari_Dataset(Dataset):
    def __init__(self, df, tokenizer, maxlen_name, maxlen_description, device):
        
        self.name = df['name'].tolist()
        self.description = df['item_description'].tolist()
        self.price = torch.tensor(df['price'].values, dtype=torch.float).to(device)
        self.maxlen_name = maxlen_name
        self.maxlen_description = maxlen_description
        self.category_name = torch.tensor(df['category_name'].astype(int).values, dtype=torch.long).to(device)
        self.brand_name = torch.tensor(df['brand_name'].astype(int).values, dtype=torch.long).to(device)
        self.numeric_array = torch.tensor(df[numeric_features].values, dtype=torch.float).to(device)
        self.gamma_bins = torch.tensor(df['gamma_bins'].values, dtype=torch.long).to(device)
        
        tokenizer_kwargs = {
            'add_special_tokens': True,
            'return_token_type_ids': False,
            'padding': 'max_length',
            'truncation': True,
            'return_attention_mask': True,
            'return_tensors': 'pt'
            }
        
        self.encoded_name = tokenizer(self.name, max_length=maxlen_name, 
                                      **tokenizer_kwargs)
        self.encoded_description = tokenizer(self.description, 
                                             max_length=maxlen_description, 
                                             **tokenizer_kwargs)
        
        for dct in (self.encoded_name, self.encoded_description):
            for k in dct.keys():
                dct[k] = dct[k].to(device)
        
    def __len__(self):
        return len(self.name)
    
    def __getitem__(self, idx):
        return ({
                'features': {
                    'encoded_name': {k: v[idx] for k,v in self.encoded_name.items()},
                    'encoded_description': {k: v[idx] for k,v in self.encoded_description.items()},
                    'category_name': self.category_name[idx],
                    'brand_name': self.brand_name[idx],
                    'x_numeric': self.numeric_array[idx,:]
                    },
                'target': self.price[idx],
                'gamma_bins': self.gamma_bins[idx]
                })
        

# %% create mercari datasets
#the maxlen description is set to be in the 90th percentile of description token
#length while trying not to be too memory intensive

maxlen_name = 20
maxlen_description = 120
train_dataset = Mercari_Dataset(train, tokenizer, maxlen_name, maxlen_description, device)
val_dataset = Mercari_Dataset(val, tokenizer, maxlen_name, maxlen_description, device)
# %% create dataloaders
batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# %% target range
#target_range = [0, train['price'].max()]
target_range = [0.,500.]
# %% create a model class with learned pdf mc layer
class learned_target_bert_model(torch.nn.Module):
    
    def __init__(self, transformer, category_embedding_dim, brand_embedding_dim,
                 numeric_dim, text_reduction_dim, struct_reduction_dim, 
                 hidden_dims, binned_gamma_params, target_range, device, 
                 mlp_activation=torch.nn.ReLU(), 
                 mc_samples=10, delta_t=torch.tensor(0.5),
                 dropout=0.1, torch_seed=123):
        
        super().__init__()
        
        #self.sentence_transformer = sentence_transformer
        self.transformer = transformer
        self.device = device
        self.category_embedding_dim = category_embedding_dim
        self.brand_embedding_dim = brand_embedding_dim
        self.numeric_dim = numeric_dim
        
        transformer_dim =\
            list(self.transformer.named_parameters())[-1][1].shape[0]
            
        self.struct_dim = category_embedding_dim + brand_embedding_dim + numeric_dim
            
        self.vector_dim = text_reduction_dim + struct_reduction_dim
            
        self.hidden_dims = [self.vector_dim] + hidden_dims

        
        #create layers
        self.text_reduction_layer = nn.Linear(2 * transformer_dim, text_reduction_dim)
        self.struct_reduction_layer = nn.Linear(self.struct_dim, struct_reduction_dim)
        self.category_embedding = nn.Embedding(n_unique_category, category_embedding_dim)
        self.brand_embedding = nn.Embedding(n_unique_brand, brand_embedding_dim)
        self.learned_target_pdf_layer = lyr.Binned_Gamma_Learned_Pdf_MLP(
            self.vector_dim, self.hidden_dims, binned_gamma_params, target_range, device,
            mc_samples=mc_samples, delta_t=delta_t, mlp_activation=mlp_activation,
            dropout=dropout, torch_seed=torch_seed)
        
        self.delta_t = self.learned_target_pdf_layer.delta_t
        self.target_range = self.learned_target_pdf_layer.target_range
        self.t_range = self.learned_target_pdf_layer.t_range
        
    
    def sentence_embed(self, encoded_sentences):
        out = self.transformer(**encoded_sentences)
        classification_out = out.last_hidden_state[:,0,:]
        #classification_out = out.pooler_output
        return classification_out
    
    def text_reduction_forward(self, encoded_name, encoded_description):
        name_vec = self.sentence_embed(encoded_name)
        desc_vec = self.sentence_embed(encoded_description)
        text_vec = torch.concat([name_vec, desc_vec], axis=1)
        text_reduction_vec = self.text_reduction_layer(text_vec)
        return text_reduction_vec
    
    def struct_reduction_forward(self, category_name, brand_name, x_numeric):
        category_emb = self.category_embedding(category_name)
        brand_emb = self.brand_embedding(brand_name)
        struct_vec = torch.concat([category_emb, brand_emb, x_numeric], axis=1)
        struct_reduction_vec = self.struct_reduction_layer(struct_vec)
        return struct_reduction_vec
    
        
    def initial_vec_forward(self, encoded_name, encoded_description, 
                            category_name, brand_name, x_numeric):
        text_reduction_vec = self.text_reduction_forward(
            encoded_name, encoded_description
        )
        struct_reduction_vec = self.struct_reduction_forward(
            category_name, brand_name, x_numeric
        )
        out_vec = torch.concat([text_reduction_vec, struct_reduction_vec], axis=1)
        return out_vec
    
            
    def forward(self, encoded_name, encoded_description, 
                            category_name, brand_name, x_numeric, t, gamma_bins):
        x = self.initial_vec_forward(encoded_name, encoded_description,
                                     category_name, brand_name, x_numeric)
        y, y_array, q_array = self.learned_target_pdf_layer(x, t, gamma_bins)
        return y, y_array, q_array
    
    def integration_forward(self, encoded_name, encoded_description, 
                            category_name, brand_name, x_numeric, t):
        x = self.initial_vec_forward(encoded_name, encoded_description,
                                     category_name, brand_name, x_numeric)
        y, y_range = self.learned_target_pdf_layer.integration_forward(x,t)
        return y, y_range


# %% define eval function #CONTINUE HERE
def gamma_model_eval(model, loss_function, dataloader, device):
        
    model.eval()
    epoch_loss = 0.0
    for data in dataloader:
        with torch.no_grad():
            y, y_array, q_array =\
                model(t=data['target'], gamma_bins=data['gamma_bins'], **data['features'])
        y, y_array, q_array = y.to(device), y_array.to(device), q_array.to(device)
        loss = loss_function(y, q_array, *y_array.T)
        batch_loss = loss.item()
        epoch_loss += batch_loss
        
    epoch_loss /= len(dataloader)
    model.train(True)
    return(epoch_loss)

# %% create model
mc_samples=50
delta_t = torch.tensor(0.5)
hidden_dims = [1024, 1024]
category_embedding_dim = 512
brand_embedding_dim = 512
text_reduction_dim = 256
struct_reduction_dim = 512
numeric_dim = len(numeric_features)
mlp_activation = torch.nn.ReLU()
dropout=0.1 
torch_seed = 123

model = learned_target_bert_model(
    transformer=bert,
    category_embedding_dim=category_embedding_dim,
    brand_embedding_dim=brand_embedding_dim,
    numeric_dim=numeric_dim,
    text_reduction_dim=text_reduction_dim,
    struct_reduction_dim=struct_reduction_dim,
    hidden_dims=hidden_dims,
    binned_gamma_params=bin_creator.binned_gamma_params,
    target_range=target_range,
    device=device,
    mlp_activation=mlp_activation,
    mc_samples=mc_samples,
    delta_t=delta_t,
    dropout=dropout,
    torch_seed=torch_seed
)

model = model.to(device)

model_n_params = sum([x.numel() for x in list(model.parameters())])
added_params = model_n_params - n_params
print(model_n_params)
print(added_params)
# %% set up training parameters
learning_rate = 1e-5
num_epochs=2
weight_decay = 1e-4
batch_update = 50
seed = 124
batch_fraction = 0.8 #fraction of batches to load in single epoch

torch.manual_seed(seed)
optimizer = torch.optim.AdamW(params=model.parameters(), 
                              lr=learning_rate, weight_decay=weight_decay)

loss_function = lf.LearnedPDFMCImportanceSampling.apply
# %% perform training loop 

torch.cuda.empty_cache()
gc.collect()

train_loss = []
eval_loss = []

t = time.time()

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
        
        #get model outputs
        y, y_array, q_array =\
            model(t=data['target'], gamma_bins=data['gamma_bins'], **data['features'])
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
            
        if i > round(len(train_dataloader) * batch_fraction):
            break
            
    epoch_loss /= len(train_dataloader)
    print('-' * 20)
    print(f'Loss after epoch {epoch+1}: {epoch_loss}')
    
    eval_epoch_loss = gamma_model_eval(model, loss_function, val_dataloader, device)
    print(f'eval loss after epoch {epoch+1}: {eval_epoch_loss}')
    print()
    
    train_loss.append(epoch_loss)
    eval_loss.append(eval_epoch_loss)
    
t = time.time() - t
print(f'training time: {t} seconds')
    

# %% scoring functions
def learned_pdf_predict(model, test_dataset, batch_size=128, batch_fraction=0.05):
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=not (batch_fraction==1))
    y = np.array([])
    y_range = None
    t = np.array([])
    #device = model.device
    for i, data in enumerate(dataloader,0):
        with torch.no_grad():
            model.eval()
            out_y, out_yrange = model.integration_forward(t=data['target'], **data['features'])
        model.train(True)
        out_y, out_yrange = out_y.detach().cpu().numpy(), out_yrange.detach().cpu().numpy()
        y = np.concatenate([y, out_y])
        y_range = np.concatenate([y_range, out_yrange], axis=0) if y_range is not None else out_yrange
        t = np.concatenate([t, data['target'].detach().cpu().numpy()])
        
        if i > round(len(dataloader) * batch_fraction):
            break
        
    return y, y_range, t

def learned_pdf_score(model, test_dataset, batch_size=128, batch_fraction=0.05,
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
    
    y, y_range, t = learned_pdf_predict(model, test_dataset, batch_size, batch_fraction)
    e_yrange = np.exp(y_range)
    Z = (e_yrange * delta_t).sum(axis=1)
    p_dist = e_yrange/Z.reshape(-1,1)
    p = np.exp(y) / Z
    t_ev = (e_yrange * delta_t * t_range).sum(axis=1) / Z
    
    return p, p_dist, t, t_ev

# %% score and evalute test set
p, p_dist, t, t_ev = learned_pdf_score(model, val_dataset, batch_fraction=1)

print(r2_score(t, t_ev)) #0.425593043329873
print(rmsle(t, t_ev)) #0.4790678534193915

# %% plot learned pdf function
def plot_dist(p_dist, t=None, tev=None, t_range=model.t_range.detach().cpu().numpy()):
        
    fig, ax = plt.subplots()
    ax.plot(t_range, p_dist)
    
    ax.set(xlabel='t', ylabel='learned target dist',
           title = 'learned target distribution',
           xlim = (0, min(2.5*max(t, tev), t_range[-1])))
    ax.grid()
    
    if t:
        plt.axvline(x=t, color='b')
    
    if tev:
        plt.axvline(x=tev, color='r')
        
    #ax.set_xlim(0, max(1.5*max(t, tev), t_range[-1]))
    
    plt.show()
    
# %%
t_range=model.t_range.detach().cpu().numpy()
# %% sample a result and show the plot

idx = random.randrange(len(p))
print(f'index: {idx}')

data = val.iloc[idx,:]

print(f'name: {data["name"]}')
print(f'item description: {data["item_description"]}')

label_vals = enc.enc.inverse_transform(pd.DataFrame(data[label_cols]).T)[0]
print(pd.Series(label_vals, index=label_cols))

print(data[['item_condition_id', 'shipping', 'price']])

scored_sample = dict({'target': t[idx], 'mean_pred': t_ev[idx]})
pprint(scored_sample)

true_val = scored_sample['target']
ev = scored_sample['mean_pred']

plot_dist(p_dist[idx], t=t[idx], tev=t_ev[idx], t_range=t_range)


