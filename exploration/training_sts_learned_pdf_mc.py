"""
training a BERT model on the STS dataset using learned PDF MC loss function
"""

import sys
import traceback
from datetime import datetime
import gc
import numpy as np
from sklearn.metrics import r2_score

from datasets import load_dataset

import torch
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

# %% create a model class with learned pdf mc layer
class sentence_transformer_mc_pdf(torch.nn.Module):
    
    def __init__(self, transformer, hidden_dims, target_range, device,
                 mc_samples=10, delta_t=torch.tensor(0.1),
                 mlp_activation=torch.nn.ReLU(), dropout=0.1, torch_seed=123):
        
        super().__init__()
        
        #self.sentence_transformer = sentence_transformer
        self.transformer = transformer
        self.device = device
        
        transformer_dim =\
            list(self.transformer.named_parameters())[-1][1].shape[0]
            
        self.learned_pdf_layer = lyr.learned_pdf_mc_mlp(
            vector_dim=transformer_dim*2, hidden_dims=hidden_dims, 
            target_range=target_range, device=device, mc_samples=mc_samples,
            delta_t=delta_t, mlp_activation=mlp_activation, dropout=dropout,
            torch_seed=torch_seed)
        
        self.delta_t = self.learned_pdf_layer.delta_t
        self.target_range = self.learned_pdf_layer.target_range
        self.t_range = self.learned_pdf_layer.t_range
    
    def sentence_embed(self, encoded_sentences):
        out = bert(**encoded_sentences)
        classification_out = out.last_hidden_state[:,0,:]
        return classification_out
        
    def st_forward(self, encoded_sentence1, encoded_sentence2):
        s1_vec = self.sentence_embed(encoded_sentence1)
        s2_vec = self.sentence_embed(encoded_sentence2)
        
        return torch.concat(
            [s1_vec, s2_vec], axis=1).to(self.device)
            
    def forward(self, encoded_sentence1, encoded_sentence2, t):
        x = self.st_forward(encoded_sentence1, encoded_sentence2)
        return self.learned_pdf_layer.forward(x, t)
    
    def integration_forward(self, encoded_sentence1, encoded_sentence2, t):
        x = self.st_forward(encoded_sentence1, encoded_sentence2)
        return self.learned_pdf_layer.integration_forward(x,t)
    
# %% define eval function
def learned_pdf_eval(model, loss_function, dataloader, device, delta_t=None,
                     target_range=None):
    
    if delta_t is None:
        try:
            delta_t = model.delta_t
        except AttributeError:
            raise ValueError('delta_t must be provided, or an attribute of model')
            
    if target_range is None:
        try:
            target_range = model.target_range
        except AttributeError:
            raise ValueError('target_range must be provided, or an attribute of model')
    
    model.eval()
    epoch_loss = 0.0
    for data in dataloader:
        with torch.no_grad():
            y, y_range = model(*data)
        y, y_range = y.to(device), y_range.to(device)
        loss = loss_function(y, target_range, *y_range.T)
        batch_loss = loss.item()
        epoch_loss += batch_loss
        
    epoch_loss /= len(dataloader)
    model.train(True)
    return(epoch_loss)

# %% create model
hidden_dims = [1024]
target_range = [0,1]
mc_samples = 60
delta_t = 0.1 
mlp_activation = torch.nn.ReLU()
dropout=0.1 
torch_seed = 123

model = sentence_transformer_mc_pdf(
    transformer=bert,
    hidden_dims=hidden_dims,
    target_range=target_range,
    device=device,
    mc_samples=mc_samples,
    delta_t=delta_t,
    mlp_activation=mlp_activation,
    dropout=dropout,
    torch_seed=torch_seed
    )

model = model.to(device)

# %% set up optimization
learning_rate = 5e-5
num_epochs=50
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
        
        #get data
        enc_sentence1, enc_sentence2, t = data
        
        #zero the gradients
        optimizer.zero_grad()
        
        #get model outputs
        y, y_range = model(enc_sentence1, enc_sentence2, t)
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
    
    test_epoch_loss = learned_pdf_eval(model, loss_function, test_dataloader, device)
    print(f'test loss after epoch {epoch+1}: {test_epoch_loss}')
    print()
    
    
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
# %% scoring functions
def learned_pdf_predict(model, test_dataset, batch_size=1024):
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    y = np.array([])
    y_range = None
    #device = model.device
    for data in dataloader:
        #for k,v in data_dict.items():
        #    if torch.is_tensor(v):
        #        data_dict[k] = v.to(device)
        with torch.no_grad():
            model.eval()
            out_y, out_yrange = model.integration_forward(*data)
        model.train(True)
        out_y, out_yrange = out_y.detach().cpu().numpy(), out_yrange.detach().cpu().numpy()
        y = np.concatenate([y, out_y])
        y_range = np.concatenate([y_range, out_yrange], axis=0) if y_range is not None else out_yrange
        
    t = test_dataset[:][2].detach().cpu().numpy()
        
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
    
    
# %%
p, p_dist, t, t_ev = learned_pdf_score(model, test_dataset)

print(r2_score(t, t_ev)) #0.027710378170013428
