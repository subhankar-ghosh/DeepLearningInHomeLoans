import config
from DataGen import DataGen
import numpy as np
import os
import pandas as pd
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve


class ModelTrainer:
    '''
    Class Model Trainer includes all function implementations for training and testing.
    Training happens as a binary classification, so it is like a one-vs-all kind of training.
    concerned_class in the constructor is the class we want to identify.
    '''
    def __init__(self, model, learning_rate, num_epochs, concerned_class, data_file_name_train, data_file_name_test, device):
        self.input_size = 110
        self.num_classes = len(config.unique_values['current_status'])
        self.learning_rate = learning_rate
        self.criterion = nn.NLLLoss(ignore_index = -1) # nn.CrossEntropyLoss()
        # self.criterion_lstm = nn.CrossEntropyLoss(ignore_index = -1)  
        self.criterion_lstm = nn.NLLLoss(ignore_index = -1)
        self.model = model
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)  
        self.num_epochs = num_epochs
        self.training_accuracy = np.zeros(shape=(self.num_epochs, 1))
        self.test_accuracy = np.zeros(shape=(self.num_epochs, 1))
        self.training_auc = np.zeros(shape=(self.num_epochs, 1))
        self.test_auc = np.zeros(shape=(self.num_epochs, 1))
        self.test_loss = []
        self.loss_values = []
        self.concerned_class = concerned_class
        self.data_file_name_train = data_file_name_train
        self.data_file_name_test = data_file_name_test
        self.device = device
        self.model = self.model.to(self.device)

    def trainFF(self, train, test):
        '''
        Training, testing and evaluation of a feed forward network
        '''
        datagen = DataGen(self.data_file_name_train, self.data_file_name_test)
        for epoch in range(self.num_epochs):
            correct = 0
            total = 0
            count = 0
            train_auc = 0
            for i, t in enumerate(train):
                x, y_true = datagen.GenerateIndependentData(t, 'train')
                x = Variable(torch.FloatTensor(x).to(self.device))
                y = Variable(torch.LongTensor(np.argmax(y_true, axis=1)).to(self.device))
                # Forward + Backward + Optimize
                self.optimizer.zero_grad()
                outputs = self.model(x)
                
                _, predicted = torch.max(outputs.data, 1)
                # print(predicted, y)
                total += y.size(0)
                
                correct += (predicted == y.data).sum().item()
                
                loss = self.criterion(outputs, y)
                # print(list(model.parameters()))
                loss.backward()
                self.optimizer.step()
                # print(list(model.parameters()))
                # print("Before sending, shape of y_true and outputs : ", y_true.shape, outputs.shape)
                
                auc_score = self.GetAUCScore(y_true, outputs.data.cpu().numpy(), self.concerned_class)
                if auc_score != 0:
                    train_auc += auc_score
                    count += 1

                if (i+1) % 200 == 0:
                    self.loss_values.append(float(loss.item()))
                    print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Accuracy: %.4f, AUC: %.4f' % (epoch+1, self.num_epochs, i+1, len(train), \
                    loss.data[0], correct/total, (train_auc/count) if count > 0 else 0))        
            self.training_auc[epoch] = float(train_auc/count)
            self.training_accuracy[epoch] = float(correct/total)
            
            total = 0
            correct = 0
            count = 0
            test_auc = 0
            for i, t in enumerate(test):
                x, y_true = datagen.GenerateIndependentData(t, 'test')
                x = Variable(torch.FloatTensor(x).to(self.device))
                y = Variable(torch.LongTensor(np.argmax(y_true, axis=1)).to(self.device))
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                _, predicted = torch.max(outputs.data, 1)
                # print(predicted, y)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                # print("Before sending, shape of y_true and outputs : ", y_true.shape, outputs.shape)
                
                auc_score = self.GetAUCScore(y_true, outputs.data.cpu().numpy(), self.concerned_class)
                if auc_score != 0:
                    count += 1
                    test_auc += auc_score
                self.test_loss.append(float(loss.item()))
            print('Test AUC: ', float(test_auc/count))
            print(self.test_loss)
            self.test_accuracy[epoch] = float(correct/total)
            self.test_auc[epoch] = float(test_auc/count)


    def trainLSTM(self, train, test):
        '''
        Training, Testing and Evaluation of LSTM.
        '''
        datagen = DataGen(self.data_file_name_train, self.data_file_name_test)
        # sftmx_layer = F.log_softmax()
        rolling_loss = 0
        loss_count = 0
        for epoch in range(self.num_epochs):
            correct = 0
            total = 0
            train_auc = 0
            count = 0
            st_time = time.time()
            for i, t in enumerate(train):
                XX, yy = datagen.DataCreatorSequence(t, 'train') # hdf_file_path
                # print('XX.shape = ', XX.shape)
                for j in range(0, XX.shape[0], 100):
                    # print('j = ', j)
                    X, y_true= XX[j:(j + 100), :, :], yy[j:(j + 100), :, :]
                    if X.shape[0] == 0:
                        continue
                    # print(X.shape, y.shape)
                    inputs = Variable(torch.FloatTensor(X).to(self.device))
                    y = np.argmax(y_true, axis=2)
                    y_oh = Variable(torch.LongTensor(y.reshape(y.shape[0]*y.shape[1])).to(self.device))
                    # print('y_oh.shape: ', y_oh.shape)
                    # print('Inputs.shape = ', inputs.shape)
                    self.optimizer.zero_grad()
                    # Forward + Backward + Optimize
                    outputs = self.model(inputs)
                    # print('y_oh.shape: ', y_oh.shape)
                    # print(y_oh)
                    # print('outputs.shape: ', outputs.shape)

                    loss = self.criterion_lstm(outputs, y_oh)   ##### Check criterion
                    loss.backward()
                    self.optimizer.step()
                    # print(outputs.shape)
                    # print('loss value = ', loss.item())
                    #print(outputs.shape)
                    y_true = y_true.reshape((y.shape[0]*y_true.shape[1], -1))
                    # print(y_true)
                    # print("Before sending, shape of y_true and outputs : ", y_true.shape, outputs.shape)
                    auc = self.GetAUCScore(y_true, outputs.data.cpu().numpy(), self.concerned_class)
                    # print(auc)
                    rolling_loss += loss.item()
                    loss_count += 1
                    if auc > 0:
                        train_auc += auc
                        count += 1
                if (i%100) == 0:
                    self.loss_values.append(float(rolling_loss/loss_count))
                    print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Roc: %.4f' % (epoch+1, self.num_epochs, \
                            i+1, len(train), float(rolling_loss/loss_count), (train_auc/count) if count > 0 else 0))
            self.training_auc[epoch] = float(train_auc/count)
            

            total = 0
            correct = 0
            count = 0
            count1 = 0
            test_auc_val1 = 0
            test_auc_val = 0
            for i, t in enumerate(test):
                XX, yy = datagen.DataCreatorSequence(t, 'test') # hdf_file_path
                # print('XX.shape = ', XX.shape)
                for j in range(0, XX.shape[0], 100):
                    # print('j = ', j)
                    X, y_true= XX[j:(j + 100), :, :], yy[j:(j + 100), :, :]
                    if X.shape[0] == 0:
                        continue
                    # print(X.shape, y.shape)
                    inputs = Variable(torch.FloatTensor(X).to(self.device))
                    y = np.argmax(y_true, axis=2)
                    y_oh = Variable(torch.LongTensor(y.reshape(y.shape[0]*y.shape[1])).to(self.device))
                    # print('y_oh.shape: ', y_oh.shape)
                    # Forward + Backward + Optimize
                    outputs = self.model(inputs)
                    loss = self.criterion_lstm(outputs, y_oh)
                    # print('y_oh.shape: ', y_oh.shape)
                    # print(y_oh)
                    # print('outputs.shape: ', outputs.shape)
                    # print(outputs.shape)
                    # print(loss.data)
                    #print(outputs.shape)
                    y_true = y_true.reshape((y.shape[0]*y_true.shape[1], -1))
                    #print(y_true.shape)
                    auc = self.GetAUCScore(y_true, outputs.data.cpu().numpy(), self.concerned_class, is_transition = False, save_results = True)
                    auc1 = self.GetAUCScore(y_true, outputs.data.cpu().numpy(), 1, is_transition = True, save_results = True)
                    # print(auc)
                    if auc1 > 0:
                        test_auc_val1 += auc1
                        count1 += 1
                    if auc > 0:
                        test_auc_val += auc
                        count += 1
                self.test_loss.append(float(loss.item()))
            print('Test AUC 46: ', float(test_auc_val/count))
            print('Test AUC 1: ', float(test_auc_val1/count1))
            print("Test Loss: ", self.test_loss)
            self.test_auc[epoch] = float(test_auc_val/count)


    def GetOutputFF(self, val_index):
        total = 0
        correct = 0
        count = 0
        test_auc = 0
        sftmx_layer = nn.Softmax()
        outputs_list = []
        for i, t in enumerate(val_index):
            x, y = datagen.GenerateData(t)
            x = Variable(torch.FloatTensor(x).to(self.device))
            y = Variable(torch.LongTensor(np.argmax(y, axis=1)).to(self.device))
            outputs = self.model(x)
            outputs_list.append(outputs)
            loss = self.criterion(outputs, y)
            
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted, y)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            auc_score = self.GetAUCScore(y_true, sftmx_layer(outputs).data.cpu().numpy(), self.concerned_class)
            if auc_score != 0:
                count += 1
                test_auc += auc_score
        return ( float(correct/total), float(test_auc/count) )


    def GetModel(self):
        return self.model   


    def accumulate_truth(self, iterable):
        '''
        Helper function to calculate AUC
        '''
        true = 0
        false = 0
        for i in iterable:
            if i:
                true += 1
            else:
                false += 1
            yield true, false


    def home_made_ROC(self, scores, targets):
        '''
        Custom code to Calculate ROC
        '''
        total_targets = sum(targets)
        length_sub_targets = len(targets) - total_targets

        scores, targets = zip(*sorted(zip(scores, targets), reverse=True))
        tprs = []
        fprs = []
        for true, false in self.accumulate_truth(targets):
            tprs.append(true / total_targets)
            fprs.append(false / length_sub_targets)
        return tprs, fprs


    def home_made_AUC(self, TPR, FPR):
        '''
        Custom code to Calculate AUC
        '''
        dFPR = np.zeros(len(FPR))
        dTPR = np.zeros(len(TPR))
        dFPR[:-1] = np.diff(FPR)
        dTPR[:-1] = np.diff(TPR)
        auc_val = np.sum(TPR * dFPR) + (np.sum(dTPR * dFPR)/2)
        if np.isnan(auc_val):
            return 0
        return auc_val


    def GetAUCScore(self, y_actual, predicted_score, concerned_class, is_transition=False, save_results=False):
        '''
        Returns AUC score given predicted score and actual y values.
        '''
        ### y_actual: one hot encoding of all classes. np array of shape n x num_of_classes
        ### predicted_score: np array of shape n x num_of_classes containing probabilities
        try:
        # print(predicted_score[:100, 0])
        # print('#################################')
        # print('########## class 0 #################')
        # print('#################################')
            if is_transition:
                ii = np.where(y_actual[:, 0] == 1)[0]
                ii = ii + 1
                ii = ii[ii < y_actual.shape[0]]
                y_actual = y_actual[ii, :]
                predicted_score = predicted_score[ii, :]
            predicted_score = predicted_score[:, concerned_class]
            y_actual = y_actual[:, concerned_class]
            valid_index = (y_actual != -1)
            y_actual = y_actual[valid_index]
            # print('Number of positive samples: ', np.sum(y_actual), 'total: ', y_actual.shape)
            # y_actual = y_actual[:, concerned_class]
            predicted_score = predicted_score[valid_index]
            if save_results:
                res = np.vstack((y_actual, predicted_score))
                np.save('res_'+str(concerned_class)+'_'+str(random.randint(0, 10000))+'.npy', res)
            # print(roc_auc_score(y_actual, predicted_score), y_actual[y_actual == 1].shape, y_actual.shape)
    #         print(y_actual[:100])
    #         print('#################################')
    #         print('########### actual class 1 ###############')
    #         print('#################################')
    #         print(predicted_score[:100,])
    #         print('#################################')
    #         print('############ class 1 ############')
    #         print('#################################')
            tprs, fprs = self.home_made_ROC(predicted_score, y_actual)
            return self.home_made_AUC(tprs, fprs)
            # return roc_auc_score(y_actual, predicted_score)
        except Exception as e:
            print("Error", e)
            # print(predicted_score)
            return 0
