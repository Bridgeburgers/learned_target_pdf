
"""
training a BERT model on the mercari price suggestion data using learned target pdf
"""

import os
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import scipy
import random
from pprint import pprint

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader

import transformers

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
# %% try to use bert
s = ['this is a sentence', 'this is a much longer sentence than the first one']
s_enc = tokenizer(s, return_tensors='pt', padding=True, truncation=True)
s_enc = {k:v.to(device) for k,v in s_enc.items()}
#s_enc = {k:v for k,v in s_enc.items() if k in ['input_ids', 'attention_mask']}

res = bert(**s_enc)
classification_res = res.last_hidden_state[:,0,:]
#classification_res = res.pooler_output
classification_res.shape
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

# %% function for cumulative percentage of items
def num_cum_percent(series, target=0.95):
    cum_percent = series.value_counts(dropna=False, normalize=True).cumsum()
    return (cum_percent < target).sum() + 1
# %% explore mercari dataset
print(list(train))
# ['train_id', 'name', 'item_condition_id', 'category_name', 
# 'brand_name', 'price', 'shipping', 'item_description']

train.isnull().sum(axis=0)
# =============================================================================
# train_id                  0
# name                      0
# item_condition_id         0
# category_name          5038
# brand_name           505941
# price                     0
# shipping                  0
# item_description          5
# =============================================================================

train['category_name'].nunique() #1266
print(num_cum_percent(train['category_name'], 0.95)) #358
print(train['category_name'].value_counts(dropna=False).head(15))
# =============================================================================
# category_name
# Women/Athletic Apparel/Pants, Tights, Leggings                 48143
# Women/Tops & Blouses/T-Shirts                                  37121
# Beauty/Makeup/Face                                             27547
# Beauty/Makeup/Lips                                             23902
# Electronics/Video Games & Consoles/Games                       21203
# Beauty/Makeup/Eyes                                             20105
# Electronics/Cell Phones & Accessories/Cases, Covers & Skins    19714
# Women/Underwear/Bras                                           17015
# Women/Tops & Blouses/Blouse                                    16295
# Women/Tops & Blouses/Tank, Cami                                16218
# Women/Dresses/Above Knee, Mini                                 16094
# Women/Jewelry/Necklaces                                        15806
# Women/Athletic Apparel/Shorts                                  15654
# Beauty/Makeup/Makeup Palettes                                  15243
# Women/Shoes/Boots                                              15038
# =============================================================================

train['brand_name'].nunique() #4525
print(num_cum_percent(train['brand_name'], 0.95)) #377
print(train['brand_name'].value_counts(dropna=False).head(15))
# =============================================================================
# brand_name
# NaN                  505941
# Nike                  43307
# PINK                  43158
# Victoria's Secret     38546
# LuLaRoe               24906
# Apple                 13823
# FOREVER 21            12075
# Nintendo              12068
# Lululemon             11600
# Michael Kors          11206
# American Eagle        10617
# Rae Dunn               9856
# Sephora                9679
# Coach                  8466
# Bath & Body Works      8250
# =============================================================================

train['item_condition_id'].value_counts(dropna=False)

# =============================================================================
# item_condition_id
# 1    512805
# 3    345469
# 2    300196
# 4     25645
# 5      1913
# =============================================================================

train['shipping'].value_counts(dropna=False) #1 if shipping is paid by seller, according to kaggle
# =============================================================================
# shipping
# 0    655572
# 1    530456
# =============================================================================

train['price'].describe()
# =============================================================================
# count    1.186028e+06
# mean     2.673895e+01
# std      3.864676e+01
# min      0.000000e+00
# 25%      1.000000e+01
# 50%      1.700000e+01
# 75%      2.900000e+01
# max      2.009000e+03
# =============================================================================

#look at the distribution of token lengths of each text field
encoded_name = tokenizer(train['name'].sample(10000).tolist(), return_tensors='pt', truncation=True,
                         max_length=500, padding='max_length', return_attention_mask=True)
name_token_count = pd.Series(encoded_name['attention_mask'].sum(axis=1))
print(name_token_count.quantile(np.arange(0.05, 1.05, 0.05)))
# =============================================================================
# 0.05     4.0
# 0.10     5.0
# 0.15     5.0
# 0.20     6.0
# 0.25     6.0
# 0.30     7.0
# 0.35     7.0
# 0.40     7.0
# 0.45     8.0
# 0.50     8.0
# 0.55     8.0
# 0.60     9.0
# 0.65     9.0
# 0.70    10.0
# 0.75    10.0
# 0.80    10.0
# 0.85    11.0
# 0.90    12.0
# 0.95    13.0
# 1.00    21.0
# =============================================================================
del encoded_name

encoded_description = tokenizer(train['item_description'].sample(10000).tolist(), 
                                return_tensors='pt', truncation=True, max_length=500, 
                                padding='max_length', return_attention_mask=True)
desc_token_count = pd.Series(encoded_description['attention_mask'].sum(axis=1))
print(desc_token_count.quantile(np.arange(0.05, 1.05, 0.05)))
# =============================================================================
# 0.05      5.0
# 0.10      5.0
# 0.15      8.0
# 0.20     10.0
# 0.25     11.0
# 0.30     13.0
# 0.35     15.0
# 0.40     17.6
# 0.45     20.0
# 0.50     23.0
# 0.55     26.0
# 0.60     29.0
# 0.65     33.0
# 0.70     38.0
# 0.75     44.0
# 0.80     54.0
# 0.85     66.0
# 0.90     86.0
# 0.95    129.0
# 1.00    309.0
# =============================================================================
del encoded_description

# %% fill nulls for all fields that contain them

for df in train, val, test_df, test_stg2_df:
    df.loc[df['item_description'].isnull(), 'item_description'] = ''
    
# %% remove zeroes in target
for df in train, val:
    df.loc[df['price']<=0., 'price'] = 0.01
    
    
# %% explore target quantile distributions

qbins = pd.qcut(train['price'], 10, labels=range(10))
train['price_bin'] = qbins

for bin in range(10):
    price_bin = train.loc[train['price_bin']==bin, 'price']
    plt.hist(price_bin, bins=30)
    plt.show()
    
plt.hist(train.loc[train['price']>400, 'price'], bins=50)
plt.show()

# fit data to gamma distribution
alpha_bar, theta_bar = utl.gamma_unbiased_estimator(train['price'].values)

z = torch.distributions.gamma.Gamma(alpha_bar, 1/theta_bar).sample([100000]) 
z = pd.Series(z)   

plt.hist(z, bins=200)
plt.show()

alpha_high, theta_high = utl.gamma_unbiased_estimator(train.loc[train['price']>=400, 'price'].values)
z = torch.distributions.gamma.Gamma(alpha_high, 1/theta_high).sample([10000]).numpy()
plt.hist(z, bins=100)
plt.show()

alpha_9, theta_9 = utl.gamma_unbiased_estimator(train.loc[train['price_bin']==9, 'price'].values)
z = torch.distributions.gamma.Gamma(alpha_9, 1/theta_9).sample([100000]).numpy()
plt.hist(z, bins=100)
plt.show()

for bin in range(10):
    alpha_hat, theta_hat = utl.gamma_unbiased_estimator(train.loc[train['price_bin']>=bin, 'price'].values)
    z = torch.distributions.gamma.Gamma(alpha_hat, 1/theta_hat).sample([100000]).numpy()
    plt.hist(z, bins=100)
    plt.title(f'fitted gamma dist for bin >= {bin}')
    plt.show()
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
                'target': self.price[idx]
                })
        

# %% create mercari datasets
#the maxlen description is set to be in the 90th percentile of description token
#length while trying not to be too memory intensive

# =============================================================================
# #OPTIONAL REDUCTION:
# train = train.sample(10000)
# val = val.sample(2000)
# =============================================================================

maxlen_name = 20
maxlen_description = 120
train_dataset = Mercari_Dataset(train, tokenizer, maxlen_name, maxlen_description, device)
val_dataset = Mercari_Dataset(val, tokenizer, maxlen_name, maxlen_description, device)
# %% create dataloaders
batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# %% write custom gamma loss function
class GammaLoss(Function):
    
    @staticmethod
    def forward(ctx, logtheta, logk, t):
        #import pdb; pdb.set_trace()
        theta = torch.clamp(torch.exp(logtheta), min=1e-6, max=1e6).to(device)
        k = torch.clamp(torch.exp(logk), min=1e-6, max=1e6).to(device)
        ctx.save_for_backward(t, theta, k)
        loss = torch.lgamma(k) + k * torch.log(theta) -\
            (k-1)*torch.log(t) + t/theta
        return loss.mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        t, theta, k = ctx.saved_tensors
        
        logtheta_grad = grad_output * (
            k - t / theta
            )
        logk_grad = grad_output * (
            k * torch.digamma(k) + k * torch.log(theta / t)
            )

        return logtheta_grad, logk_grad, None

# %% create a model class with learned pdf mc layer
class gamma_bert_model(torch.nn.Module):
    
    def __init__(self, transformer, category_embedding_dim, brand_embedding_dim,
                 numeric_dim, text_reduction_dim, struct_reduction_dim, 
                 hidden_dims, device, mlp_activation=torch.nn.ReLU(), 
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
            
        hidden_dims = [self.vector_dim] + hidden_dims
        mlp_layers = [
            layer
            for prev_num, num in zip(hidden_dims[0:-1], hidden_dims[1:])
            for layer in [nn.Linear(prev_num, num), nn.Dropout(dropout), mlp_activation]
            ]
        
        #create layers
        self.text_reduction_layer = nn.Linear(2 * transformer_dim, text_reduction_dim)
        self.struct_reduction_layer = nn.Linear(self.struct_dim, struct_reduction_dim)
        self.category_embedding = nn.Embedding(n_unique_category, category_embedding_dim)
        self.brand_embedding = nn.Embedding(n_unique_brand, brand_embedding_dim)
        self.mlp = nn.Sequential(*mlp_layers)
        self.logtheta_layer = nn.Linear(hidden_dims[-1], 1)
        self.logk_layer = nn.Linear(hidden_dims[-1], 1) 
        
    
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
    
    def mlp_forward(self, x):
        v = self.mlp(x)
        logtheta = self.logtheta_layer(v)
        logk = self.logk_layer(v)
        return logtheta.squeeze(-1), logk.squeeze(-1)
            
    def forward(self, encoded_name, encoded_description, 
                            category_name, brand_name, x_numeric):
        x = self.initial_vec_forward(encoded_name, encoded_description,
                                     category_name, brand_name, x_numeric)
        logtheta, logk = self.mlp_forward(x)
        return logtheta, logk


# %% define eval function
def gamma_model_eval(model, loss_function, dataloader, device):
        
    model.eval()
    epoch_loss = 0.0
    for data in dataloader:
        with torch.no_grad():
             logtheta, logk = model(**data['features'])
        logtheta, logk = logtheta.to(device), logk.to(device)
        loss = loss_function(logtheta, logk, data['target'])
        batch_loss = loss.item()
        epoch_loss += batch_loss
        
    epoch_loss /= len(dataloader)
    model.train(True)
    return(epoch_loss)

# %% create model
hidden_dims = [1024, 1024]
category_embedding_dim = 512
brand_embedding_dim = 512
text_reduction_dim = 256
struct_reduction_dim = 512
numeric_dim = len(numeric_features)
mlp_activation = torch.nn.ReLU()
dropout=0.1 
torch_seed = 123

model = gamma_bert_model(
    transformer=bert,
    category_embedding_dim=category_embedding_dim,
    brand_embedding_dim=brand_embedding_dim,
    numeric_dim=numeric_dim,
    text_reduction_dim=text_reduction_dim,
    struct_reduction_dim=struct_reduction_dim,
    hidden_dims=hidden_dims,
    device=device,
    mlp_activation=mlp_activation,
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

loss_function = GammaLoss.apply
# %% perform training loop 

torch.cuda.empty_cache()
gc.collect()

train_loss = []
eval_loss = []

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
        logtheta, logk = model(**data['features'])
        logtheta, logk = logtheta.to(device), logk.to(device)
        
        #compute loss
        loss = loss_function(logtheta, logk, data['target'])
        
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
    
# %% check if model weights are changing
list(model.transformer.named_parameters())[0]

# =============================================================================
# ('embeddings.word_embeddings.weight',
#  Parameter containing:
#  tensor([[-0.0166, -0.0666, -0.0163,  ..., -0.0200, -0.0514, -0.0264],
#          [-0.0132, -0.0673, -0.0161,  ..., -0.0227, -0.0554, -0.0260],
#          [-0.0176, -0.0709, -0.0144,  ..., -0.0246, -0.0596, -0.0232],
#          ...,
#          [-0.0231, -0.0588, -0.0105,  ..., -0.0195, -0.0262, -0.0212],
#          [-0.0490, -0.0561, -0.0047,  ..., -0.0107, -0.0180, -0.0219],
#          [-0.0065, -0.0915, -0.0025,  ..., -0.0151, -0.0504,  0.0460]],
#         device='cuda:0', requires_grad=True))
# =============================================================================

list(model.transformer.named_parameters())[40]
# =============================================================================
# ('transformer.layer.2.attention.v_lin.weight',
#  Parameter containing:
#  tensor([[ 0.0677,  0.0032, -0.0437,  ..., -0.0605,  0.0270,  0.0181],
#          [ 0.0533,  0.0187, -0.0389,  ..., -0.0410, -0.0598, -0.0240],
#          [ 0.0196, -0.0092,  0.0391,  ...,  0.0517,  0.0742, -0.0281],
#          ...,
#          [ 0.0087, -0.0422, -0.0057,  ..., -0.0351, -0.0132, -0.0111],
#          [ 0.0274,  0.0053, -0.0396,  ...,  0.0278, -0.0205, -0.0394],
#          [-0.0118, -0.0062, -0.0310,  ..., -0.0253,  0.1374,  0.0102]],
#         device='cuda:0', requires_grad=True))
# =============================================================================

list(model.transformer.named_parameters())[99][1][700:767]
# =============================================================================
# tensor([-0.0703, -0.0584, -0.0494, -0.0784, -0.0714,  0.0137,  0.0232,  0.0267,
#         -0.0039, -0.0370, -0.0091, -0.0236,  0.0245,  0.0070, -0.0248, -0.0062,
#         -0.0658,  0.0393,  0.0427,  0.0395, -0.0133,  0.0119, -0.0266, -0.1030,
#         -0.0200,  0.0216,  0.0195, -0.1055, -0.1045, -0.0668, -0.0149,  0.0451,
#         -0.0452,  0.0075, -0.0709, -0.0118, -0.0248, -0.0092, -0.0127, -0.0255,
#         -0.0579,  0.0298,  0.0318, -0.1366, -0.1140, -0.0455, -0.0082, -0.0656,
#         -0.0117, -0.0257,  0.0433, -0.0170, -0.0279, -0.0658, -0.0033,  0.0005,
#         -0.1117,  0.0124,  0.0482, -0.0089, -0.0823, -0.0183, -0.0122, -0.0525,
#          0.0075, -0.0112,  0.0215], device='cuda:0', grad_fn=<SliceBackward0>)
# =============================================================================

list(model.parameters())[100]

# =============================================================================
# Parameter containing:
# tensor([[-0.0104,  0.0008, -0.0127,  ..., -0.0024,  0.0133,  0.0099],
#         [-0.0129, -0.0045, -0.0157,  ...,  0.0042,  0.0037,  0.0219],
#         [-0.0082, -0.0241,  0.0085,  ...,  0.0091, -0.0166, -0.0149],
#         ...,
#         [ 0.0083,  0.0027,  0.0246,  ..., -0.0109, -0.0051, -0.0075],
#         [-0.0111, -0.0112, -0.0009,  ...,  0.0209, -0.0198,  0.0123],
#         [ 0.0212,  0.0104,  0.0072,  ..., -0.0192,  0.0139,  0.0235]],
#        device='cuda:0', requires_grad=True)
# =============================================================================
# %% scoring function
def score(model, test_dataset, batch_size=128):
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    logtheta = np.array([])
    logk = np.array([])
    for data in dataloader:
        with torch.no_grad():
            model.train = False
            out_logtheta, out_logk = model(**data['features'])
        model.train = True
        out_logtheta, out_logk = out_logtheta.detach().cpu().numpy(), out_logk.detach().cpu().numpy()
        logtheta = np.concatenate([logtheta, out_logtheta])
        logk = np.concatenate([logk, out_logk])
        
    t = test_dataset[:]['target'].detach().cpu().numpy()
        
    return logtheta, logk, t

def score_values(model, test_dataset, batch_size=128):
    logtheta, logk, t = score(model, test_dataset, batch_size)
    theta = np.exp(logtheta)
    k = np.exp(logk)
    mean = k * theta
    mode = np.where(k>=1, (k-1)*theta, 0)
    std = np.sqrt(k * theta**2)
    
    return pd.DataFrame({
        'theta': theta,
        'k': k,
        'target': t,
        'mean_pred': mean,
        'mode_pred': mode,
        'std_pred': std
        })

# %% score and evalute test set
scored_val = score_values(model, val_dataset)

print(r2_score(scored_val['target'], scored_val['mean_pred'])) #0.5730158691342936
print(rmsle(scored_val['target'], scored_val['mean_pred'])) #0.43308816626818003
print(rmsle(scored_val['target'], scored_val['mode_pred'])) #0.47505032361485783

# %% plot gamma function
def plot_gamma(theta, k, true_val=None, ev=None, x_min=0.0, x_max=20, x_step=0.01):
    
    x = np.arange(x_min, x_max+x_step, x_step)
    y = x**(k-1) * np.exp(-x / theta) / scipy.special.gamma(k) / theta**k
    
    fig, ax = plt.subplots()
    ax.plot(x,y)
    
    ax.set(xlabel='x', ylabel=f'gamma(x; {round(k,2)}, {round(theta,2)})',
           title = f'Gamma Distribution with k={round(k,2)}, theta={round(theta,2)}')
    ax.grid()
    
    if true_val:
        plt.axvline(x=true_val, color='b')
        
    if ev:
        plt.axvline(x=ev, color='r')
    
    plt.show()
    
# %% sample a result and show the plot

idx = random.randrange(len(val))
print(f'index: {idx}')

data = val.iloc[idx,:]

print(f'name: {data["name"]}')
print(f'item description: {data["item_description"]}')

label_vals = enc.enc.inverse_transform(pd.DataFrame(data[label_cols]).T)[0]
print(pd.Series(label_vals, index=label_cols))

print(data[['item_condition_id', 'shipping', 'price']])

scored_sample = scored_val.iloc[idx,:]
pprint(scored_sample)

true_val = scored_sample['target']
ev = scored_sample['mean_pred']

plot_gamma(scored_sample['theta'], scored_sample['k'], 
           true_val=true_val, ev=ev, 
           x_min = max(min(0.8*true_val, ev - 2*scored_sample['std_pred']),0),
           x_max = max(1.2*true_val, ev + 2*scored_sample['std_pred'])
)