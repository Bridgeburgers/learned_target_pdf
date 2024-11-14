# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, RobertaModel
import torch


# %% load data
dir = 'D:/Documents/Data/kaggle_readability_data/'
train_file = 'train.csv'
test_file = 'test.csv'

train_df = pd.read_csv(dir + train_file)
test_df = pd.read_csv(dir + test_file)
sample_submission = pd.read_csv(dir + 'sample_submission.csv')

train_df = train_df[['id', 'excerpt', 'target', 'standard_error']]

# %% split train into train/val sets
train_df, val_df = train_test_split(train_df, test_size=int(0.2*len(train_df)), random_state=123)

# %% load roberta-base

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
model = RobertaModel.from_pretrained("FacebookAI/roberta-base")

#inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
#outputs = model(**inputs)

#last_hidden_states = outputs.last_hidden_state

# %%
