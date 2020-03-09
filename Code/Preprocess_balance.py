# import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import config

folder = '../Data/'
hdf_file_path = os.path.join(folder, 'Loan_Data.h5')
hdf_path_balanced = os.path.join(folder, 'Loan_Data_balanced.h5')

key_len = 1600
random.seed = 448
rand_ind = random.sample([i for i in range(int(key_len/2))], int(key_len/2))

train_lim = int(0.7*int(key_len/2))
test_lim = train_lim + int(0.25*int(key_len/2))
valid_lim = test_lim + int((1 - train_lim - test_lim)*int(key_len/2))
train, test, valid = rand_ind[:train_lim], rand_ind[train_lim: test_lim], rand_ind[test_lim:]

st = time.time()
index = 0
max_size = 0
df_out = pd.DataFrame()
lookup_out = pd.DataFrame()
for t in train:
    df = pd.read_hdf(hdf_file_path, 'X_' + str(t))
    lookup = pd.read_hdf(hdf_file_path, 'lookup_' + str(t))
    df.current_status = df.current_status.astype('category', categories=config.unique_values['current_status'])
    df.state = df.state.astype('category', categories=config.unique_values['unique_states'])
    for _, l in lookup.iterrows():
        df_each = df.iloc[l[1] : l[2], :]
        if(len(df_each.current_status.unique()) > 1):
            df_out = df_out.append(df_each)
            lookup_out = lookup_out.append(l)
            max_size = max(max_size, l[2] - l[1])
            index += 1
            if((index + 1) % 3000 == 0):
                df_out.to_hdf(hdf_path_balanced, key='X_'+str(int(index/3000)), format='t', complevel=9)
                lookup_out.to_hdf(hdf_path_balanced, key='lookup_'+str(int(index/3000)), format='t', complevel=9)
                df_out = pd.DataFrame()
                lookup_out = pd.DataFrame()
print('train over')
for t in test:
    df = pd.read_hdf(hdf_file_path, 'X_' + str(t))
    lookup = pd.read_hdf(hdf_file_path, 'lookup_' + str(t))
    df.current_status = df.current_status.astype('category', categories=config.unique_values['current_status'])
    df.state = df.state.astype('category', categories=config.unique_values['unique_states'])
    for _, l in lookup.iterrows():
        df_each = df.iloc[l[1] : l[2], :]
        if(len(df_each.current_status.unique()) > 1):
            df_out = df_out.append(df_each)
            lookup_out = lookup_out.append(l)
            max_size = max(max_size, l[2] - l[1])
            index += 1
            if((index + 1) % 3000 == 0):
                df_out.to_hdf(hdf_path_balanced, key='X_'+str(int(index/3000)), format='t', complevel=9)
                lookup_out.to_hdf(hdf_path_balanced, key='lookup_'+str(int(index/3000)), format='t', complevel=9)
                df_out = pd.DataFrame()
                lookup_out = pd.DataFrame()
print('test over')
for t in valid:
    df = pd.read_hdf(hdf_file_path, 'X_' + str(t))
    lookup = pd.read_hdf(hdf_file_path, 'lookup_' + str(t))
    df.current_status = df.current_status.astype('category', categories=config.unique_values['current_status'])
    df.state = df.state.astype('category', categories=config.unique_values['unique_states'])
    for _, l in lookup.iterrows():
        df_each = df.iloc[l[1] : l[2], :]
        if(len(df_each.current_status.unique()) > 1):
            df_out = df_out.append(df_each)
            lookup_out = lookup_out.append(l)
            max_size = max(max_size, l[2] - l[1])
            index += 1
            if((index + 1) % 3000 == 0):
                df_out.to_hdf(hdf_path_balanced, key='X_'+str(int(index/3000)), format='t', complevel=9)
                lookup_out.to_hdf(hdf_path_balanced, key='lookup_'+str(int(index/3000)), format='t', complevel=9)
                df_out = pd.DataFrame()
                lookup_out = pd.DataFrame()
print('valid over')
df_out.to_hdf(hdf_path_balanced, key='X_'+str(int(index/3000)), format='t', complevel=9)
lookup_out.to_hdf(hdf_path_balanced, key='lookup_'+str(int(index/3000)), format='t', complevel=9)
et = time.time()
print('time taken: ', et - st)
print('number of dfs: ', index)
print('max size: ', max_size)