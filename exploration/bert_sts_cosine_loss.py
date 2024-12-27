"""
training a BERT model on the STS dataset using learned PDF MC loss function
"""

import sys
import traceback
from datetime import datetime
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from datasets import load_dataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import transformers

import functions.layers as lyr
import functions.loss_function as lf
import functions.scoring as score

# %% You can specify any Hugging Face pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "distilbert-base-uncased"
train_batch_size = 16
num_epochs = 4
maxlen=128
output_dir = (
    "D:/Models/sts_models/" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# %% torch device
device = torch.device('cuda:0')


# %% load BERT
bert = transformers.AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

bert = bert.to(device)

# %% use bert and tokenizer
sentences = ['this is sentence', 'this is also another sentence with more words in it']
encoded_sentences = tokenizer(
    sentences, 
    padding='max_length', 
    max_length=6, truncation=True, 
    return_tensors='pt', 
    return_attention_mask=True)

for k in encoded_sentences.keys():
    encoded_sentences[k] = encoded_sentences[k].to(device)


out = bert(**encoded_sentences)
classification_out = out.last_hidden_state[:,0,:]
# %% 2. Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
train_dataset_stsb = load_dataset("sentence-transformers/stsb", split="train")
eval_dataset_stsb = load_dataset("sentence-transformers/stsb", split="validation")
test_dataset_stsb = load_dataset("sentence-transformers/stsb", split="test")

# %% create dataset class
class STS_Dataset(Dataset):
    def __init__(self, sentence1, sentence2, t, tokenizer, maxlen, device):
        
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        #self.t = 2 * torch.tensor(t, dtype=torch.float).to(device) - 1
        self.t = torch.tensor(t, dtype=torch.float).to(device)
        self.maxlen = maxlen
        
        tokenizer_kwargs = {
            'add_special_tokens': True,
            'max_length': self.maxlen,
            'return_token_type_ids': False,
            'padding': 'max_length',
            'truncation': True,
            'return_attention_mask': True,
            'return_tensors': 'pt'
            }
        
        self.encoded1 = tokenizer(sentence1, **tokenizer_kwargs)
        self.encoded2 = tokenizer(sentence2, **tokenizer_kwargs)
        
        for dct in (self.encoded1, self.encoded2):
            for k in dct.keys():
                dct[k] = dct[k].to(device)
        
    def __len__(self):
        return len(self.encoded1['input_ids'])
    
    def __getitem__(self, idx):
        return ({'input_ids': self.encoded1['input_ids'][idx], 
                 'attention_mask': self.encoded1['attention_mask'][idx]},
                {'input_ids': self.encoded2['input_ids'][idx], 
                 'attention_mask': self.encoded2['attention_mask'][idx]},
                self.t[idx]
                )
        

# %% create sts datasets
train_dataset = STS_Dataset(train_dataset_stsb['sentence1'],
                            train_dataset_stsb['sentence2'],
                            train_dataset_stsb['score'],
                            tokenizer, maxlen, device)

eval_dataset = STS_Dataset(eval_dataset_stsb['sentence1'],
                            eval_dataset_stsb['sentence2'],
                            eval_dataset_stsb['score'],
                            tokenizer, maxlen, device)

test_dataset = STS_Dataset(test_dataset_stsb['sentence1'],
                            test_dataset_stsb['sentence2'],
                            test_dataset_stsb['score'],
                            tokenizer, maxlen, device)
# %% create dataloaders
batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# %% dot product
def dot(x1, x2):
    return (x1 * x2).sum(axis=1)

# %% create a model class with learned pdf mc layer
class sentence_transformer_bert(torch.nn.Module):
    
    def __init__(self, transformer, device, torch_seed=123):
        
        super().__init__()
        
        #self.sentence_transformer = sentence_transformer
        self.transformer = transformer
        self.device = device
        
        self.transformer_dim =\
            list(self.transformer.named_parameters())[-1][1].shape[0]
    
    def sentence_embed(self, encoded_sentences):
        out = bert(**encoded_sentences)
        classification_out = out.last_hidden_state[:,0,:]
        return classification_out    
            
    def forward(self, encoded_sentence1, encoded_sentence2):
        x1 = self.sentence_embed(encoded_sentence1)
        x2 = self.sentence_embed(encoded_sentence2)
        
        return x1, x2
        #return dot(x1, x2) / (dot(x1,x1) * dot(x2,x2)).sqrt()
        
    def dot_forward(self, encoded_sentence1, encoded_sentence2):
        x1, x2 = self.forward(encoded_sentence1, encoded_sentence2)
        return dot(x1, x2) / (dot(x1,x1) * dot(x2,x2)).sqrt()
    
    
# %% cosine loss function

class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        
    def forward(self, x1, x2, target):
        cos_sim = dot(x1, x2) / (dot(x1,x1) * dot(x2,x2)).sqrt()
        return ((cos_sim - target)**2).mean()
# %% define eval function
def learned_pdf_eval(model, loss_function, dataloader, device):
    
    
    model.eval()
    epoch_loss = 0.0
    for data in dataloader:
        input1, input2, t = data
        with torch.no_grad():
            x1, x2 = model(input1, input2)
        x1, x2 = x1.to(device), x2.to(device)
        loss = loss_function(x1, x2, t)
        batch_loss = loss.item()
        epoch_loss += batch_loss
        
    epoch_loss /= len(dataloader)
    model.train(True)
    return(epoch_loss)

# %% create model

torch_seed = 123

model = sentence_transformer_bert(bert, device, torch_seed)

model = model.to(device)

# %% set up optimization
learning_rate = 5e-5
num_epochs=15
weight_decay = 1e-4
batch_update = 10
seed = 124

torch.manual_seed(seed)
optimizer = torch.optim.AdamW(params=model.parameters(), 
                              lr=learning_rate, weight_decay=weight_decay)

#loss_function = torch.nn.CosineEmbeddingLoss()
loss_function = CosineLoss()
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
        
        #get data
        enc_sentence1, enc_sentence2, t = data
        
        #zero the gradients
        optimizer.zero_grad()
        
        #get model outputs
        x1, x2 = model(enc_sentence1, enc_sentence2)
        x1, x2 = x1.to(device), x2.to(device)
        
        #compute loss
        loss = loss_function(x1, x2, t)
        
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
    
    eval_epoch_loss = learned_pdf_eval(model, loss_function, eval_dataloader, device)
    print(f'eval loss after epoch {epoch+1}: {eval_epoch_loss}')
    print()
    
    train_loss.append(epoch_loss)
    eval_loss.append(eval_epoch_loss)
    

 # %% scoring functions
def cos_predict(model, test_dataset, batch_size=2048):
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    y = np.array([])
    

    for data in dataloader:
        s1, s2, t = data
        with torch.no_grad():
            model.eval()
            y_batch = model.dot_forward(s1, s2)
        model.train(True)
        y_batch = y_batch.detach().cpu().numpy()
        y = np.concatenate([y, y_batch])
        
    t = test_dataset[:][2].detach().cpu().numpy()
        
    return pd.DataFrame({'y': y, 't': t})

    
# %%
pred_df = cos_predict(model, test_dataset)
pred_df['sentence1'] = test_dataset_stsb['sentence1']
pred_df['sentence2'] = test_dataset_stsb['sentence2']

print(r2_score(pred_df['t'], pred_df['y'])) #0.36715105465128484


# %% 
def score_two_sentences(s1, s2):
    tokenizer_kwargs = {
        'add_special_tokens': True,
        'max_length': 128,
        'return_token_type_ids': False,
        'padding': 'max_length',
        'truncation': True,
        'return_attention_mask': True,
        'return_tensors': 'pt'
        }

    
    encoded1 = tokenizer([s1], **tokenizer_kwargs).to(device)
    encoded2 = tokenizer([s2], **tokenizer_kwargs).to(device)
    return model.dot_forward(encoded1, encoded2).detach().cpu().numpy()[0]


# %%
s1 = 'I am going to the hospital'
s2 = 'am I going to the hospital?'

score_two_sentences(s1, s2)