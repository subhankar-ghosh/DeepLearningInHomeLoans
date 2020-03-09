import config
import pandas as pd
import os
import numpy as np


class DataGen:
    def __init__(self, train_file_name, test_file_name):
        '''
        file_name: 'Loan_Data.h5'
        '''
        self.folder = '../Data/'
        self.hdf_file_path_train = os.path.join(self.folder, train_file_name)
        self.hdf_file_path_test = os.path.join(self.folder, test_file_name)
        self.hdf_file_path = ''

    def GenerateIndependentData(self, t, file_type):
        if file_type == 'train':
            self.hdf_file_path = self.hdf_file_path_train
        else:
            self.hdf_file_path = self.hdf_file_path_test
        df = pd.read_hdf(self.hdf_file_path, 'X_' + str(t))
        lookup = pd.read_hdf(self.hdf_file_path, 'lookup_' + str(t))
        df.loc[(df.zero_balance == 1) == True, 'current_status'] = '46'
        df.current_UPB = pd.to_numeric(df['current_UPB'], errors='coerce')
        df.mi_recovery = pd.to_numeric(df['mi_recovery'], errors='coerce')
        df.non_mi_recovery = pd.to_numeric(df['non_mi_recovery'], errors='coerce')
        df.net_sales_procedees = pd.to_numeric(df['net_sales_procedees'], errors='coerce')
        df.modification_cost = pd.to_numeric(df['modification_cost'], errors='coerce')
        df.remaining_month = pd.to_numeric(df['remaining_month'], errors='coerce')
        df.current_status = df.current_status.astype('category', categories=config.unique_values['current_status'])
        df.state = df.state.astype('category', categories=config.unique_values['unique_states'])
        y = pd.get_dummies(df.current_status).values
        df = pd.get_dummies(df, columns=['state'])
        X = df[config.column_list['select_columns']]
        X = X.values
        X[np.isnan(X)]=-1
        return (X, y)

    def DataCreatorSequence(self, t, file_type, max_len=135):
        if file_type == 'train':
            self.hdf_file_path = self.hdf_file_path_train
        else:
            self.hdf_file_path = self.hdf_file_path_test
        df = pd.read_hdf(self.hdf_file_path, 'X_' + str(t))
        lookup = pd.read_hdf(self.hdf_file_path, 'lookup_' + str(t)).values
        df.loc[(df.zero_balance == 1) == True, 'current_status'] = '46'
        df.current_UPB = pd.to_numeric(df['current_UPB'], errors='coerce')
        df.mi_recovery = pd.to_numeric(df['mi_recovery'], errors='coerce')
        df.non_mi_recovery = pd.to_numeric(df['non_mi_recovery'], errors='coerce')
        df.net_sales_procedees = pd.to_numeric(df['net_sales_procedees'], errors='coerce')
        df.modification_cost = pd.to_numeric(df['modification_cost'], errors='coerce')
        df.remaining_month = pd.to_numeric(df['remaining_month'], errors='coerce')

        df.current_status = df.current_status.astype('category', categories=config.unique_values['current_status'])
        # df.loc[df.current_status != '1', 'current_status'] = '0'
        # df.current_status = df.current_status.astype('category', categories=['0', '1'])
        df.state = df.state.astype('category', categories=config.unique_values['unique_states'])
        y = pd.get_dummies(df.current_status).values
        df = pd.get_dummies(df, columns=['state'])
        X = df[config.column_list['select_columns']].values
        # print(X.dtypes)
        X[np.isnan(X)]=-1
        
        batch_data = np.zeros((lookup.shape[0], max_len, X.shape[1])) # batch, seq, features
        batch_labels = -1*np.ones((lookup.shape[0], max_len, y.shape[1]))
        
        for ind in range(lookup.shape[0]):
            sent = X[ lookup[ind, 1]:lookup[ind, 2], :]
            label = y[ lookup[ind, 1]:lookup[ind, 2], :]
            curr_len = sent.shape[0]
            batch_data[ind, :curr_len, :] = sent
            batch_labels[ind, :curr_len, :] = label
            
        return (batch_data, batch_labels)
