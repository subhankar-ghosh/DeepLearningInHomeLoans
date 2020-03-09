# import h5py
import config
import matplotlib.pyplot as plt
from Models import FeedForward, FeedForward5, LogisticRegression, LSTMPredictor, BatchOneLayerLSTM, BatchOneLayerGRU
from ModelTrainer import ModelTrainer
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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve

def main():
    # hdf_file_path: Loan_Data
    # Create Train, test, validation set
    # key_len = 1635
    # random.seed = 448
    # rand_ind = [i for i in range(key_len)]
    # train_lim = int(0.7*key_len)
    # test_lim = train_lim + int(0.25*key_len)
    # valid_lim = test_lim + int((1 - train_lim - test_lim)*key_len)
    # train, test, valid = rand_ind[:train_lim], rand_ind[train_lim: test_lim], rand_ind[test_lim:]

    train_len = 1830
    test_len = 700
    train = [i for i in range(train_len)]
    test = [i for i in range(test_len)]
    random.shuffle(test)
    random.shuffle(train)
    # print('train indices: ', train)
    # print('test indices: ', test)
    # Model Parameters
    INPUT_SIZE = 111
    HIDDEN_LAYER1 = 90
    HIDDEN_LAYER2 = 70
    HIDDEN_LAYER3 = 60
    HIDDEN_LAYER4 = 50
    HIDDEN_LAYER_LSTM = 128
    NUM_CLASSES = len(config.unique_values['current_status']) # 45
    # Training paramters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    CONCERNED_CLASS = 43 # 1
    DATA_FILE_TEST = 'Loan_Data_Test5.h5'
    DATA_FILE_TRAIN = 'Loan_Data_Train5.h5'

    ################# CHANGE THIS ################
    model_name = 'LSTM' # 'FF' # 'LogisticRegression'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model Selection
    if model_name == 'GRU':
        model = BatchOneLayerGRU(INPUT_SIZE, HIDDEN_LAYER_LSTM, NUM_CLASSES, device)
        print('Learning rate, number_of_epochs, concerned_class, algo, hidden_layers::', LEARNING_RATE, NUM_EPOCHS, CONCERNED_CLASS, 'GRU', HIDDEN_LAYER_LSTM)

    if model_name == 'FF':
        model = FeedForward5(INPUT_SIZE, HIDDEN_LAYER1, HIDDEN_LAYER2, HIDDEN_LAYER3, HIDDEN_LAYER4, NUM_CLASSES)
        # model = FeedForward(INPUT_SIZE, HIDDEN_LAYER1, HIDDEN_LAYER2, NUM_CLASSES)
        print('Learning rate, number_of_epochs, concerned_class, algo, hidden_layers::', LEARNING_RATE, NUM_EPOCHS, CONCERNED_CLASS, 'FF', HIDDEN_LAYER1, HIDDEN_LAYER2)

    if model_name == 'LSTM':
        model = BatchOneLayerLSTM(INPUT_SIZE, HIDDEN_LAYER_LSTM, NUM_CLASSES, device)
        print('Learning rate, number_of_epochs, concerned_class, algo, hidden_layers::', LEARNING_RATE, NUM_EPOCHS, CONCERNED_CLASS, 'LSTM', HIDDEN_LAYER_LSTM)

    if model_name == 'LogisticRegression':
        model = LogisticRegression(INPUT_SIZE, NUM_CLASSES)
        print('Learning rate, number_of_epochs, concerned_class, algo, hidden_layers::', LEARNING_RATE, NUM_EPOCHS, CONCERNED_CLASS, 'Logistic Regression', '-')

    # Create object of ModelTrainer
    trainer = ModelTrainer(model, LEARNING_RATE, NUM_EPOCHS, CONCERNED_CLASS, DATA_FILE_TRAIN, DATA_FILE_TEST, device)

    # Call train function based on model
    if model_name == 'LSTM' or model_name == 'GRU':
        trainer.trainLSTM(train, test)
    else:
        trainer.trainFF(train, test)

    print("training AUC: ", trainer.training_auc)
    print("testing AUC: ", trainer.test_auc)
    if model_name != 'LSTM' and model_name != 'GRU':
        print("training accuracy: ", trainer.training_accuracy)
        print("testing accuracy: ", trainer.test_accuracy)
    print("loss: ", trainer.loss_values)

if __name__ == "__main__":
    main()
