import config
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
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        init.xavier_normal(self.linear.weight)
    
    def forward(self, x):
        out = self.linear(x)
        return F.log_softmax( out, dim=1 )

class BatchOneLayerGRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, tagset_size, device):
        super(BatchOneLayerGRU, self).__init__()
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.device = device
        self.hidden = 0 # self.init_hidden()

    def init_hidden(self, minibatch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.FloatTensor(1, minibatch_size, self.hidden_dim).fill_(0).to(self.device),
                torch.FloatTensor(1, minibatch_size, self.hidden_dim).fill_(0).to(self.device))

    def forward(self, x):
        # seq_len = x.shape[1]
        batch_size = x.shape[0]
        # print(x.shape)
        self.hidden = self.init_hidden(batch_size)
        # x.view(seq_len, 1, -1)
        gru_out, self.hidden = self.gru(x, self.hidden)
        # print(lstm_out.shape)
        gru_out = gru_out.contiguous().view(-1, gru_out.shape[2])
        tag_space = self.hidden2tag(gru_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        # print(tag_scores.shape)
        return tag_scores


class BatchOneLayerLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, tagset_size, device):
        super(BatchOneLayerLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        print('LSTM Model with ', self.hidden_dim, ' hidden units ', input_dim, ' input dimensions and dropout of ', 0)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.device = device
        self.hidden = 0 # self.init_hidden()

    def init_hidden(self, minibatch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.FloatTensor(1, minibatch_size, self.hidden_dim).fill_(0).to(self.device),
                torch.FloatTensor(1, minibatch_size, self.hidden_dim).fill_(0).to(self.device))

    def forward(self, x):
        # seq_len = x.shape[1]
        batch_size = x.shape[0]
        # print(x.shape)
        self.hidden = self.init_hidden(batch_size)
        # x.view(seq_len, 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # print(lstm_out.shape)
        lstm_out = lstm_out.contiguous().view(-1, lstm_out.shape[2])
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        # print(tag_scores.shape)
        return tag_scores


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, tagset_size, device):
        super(LSTMPredictor, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.FloatTensor(1, 1, self.hidden_dim).fill_(0),
                torch.FloatTensor(1, 1, self.hidden_dim).fill_(0))

    def forward(self, x):
        seq_len = x.shape[1]
        lstm_out, self.hidden = self.lstm(x.view(seq_len, 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.softmax(tag_space, dim=2)
        return tag_scores

class FeedForward5(nn.Module):
    def __init__(self, input_size, hidden_layer1, hidden_layer2, hidden_layer3, hidden_layer4, num_classes):
        super(FeedForward5, self).__init__()
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_size, hidden_layer1)
        init.xavier_normal(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_layer1)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        init.xavier_normal(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_layer2)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_layer2, hidden_layer3)
        init.xavier_normal(self.fc3.weight)
        self.bn3 = nn.BatchNorm1d(num_features=hidden_layer3)
         
        self.fc4 = nn.Linear(hidden_layer3, hidden_layer4)
        init.xavier_normal(self.fc4.weight)
        self.bn4 = nn.BatchNorm1d(num_features=hidden_layer4)
        
        self.fc5 = nn.Linear(hidden_layer4, num_classes)
        init.xavier_normal(self.fc5.weight)
        self.bn5 = nn.BatchNorm1d(num_features=num_classes)
    

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu2(out)

        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu2(out)

        out = self.fc5(out)
        out = self.bn5(out)
        return F.log_softmax( out, dim=1 )

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_layer1, hidden_layer2, num_classes):
        super(FeedForward, self).__init__()
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_size, hidden_layer1)
        init.xavier_normal(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_layer1)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        init.xavier_normal(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_layer2)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_layer2, num_classes)
        init.xavier_normal(self.fc3.weight)
        self.bn3 = nn.BatchNorm1d(num_features=num_classes)
    

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.bn3(out)
        return F.log_softmax( out, dim=1 )
