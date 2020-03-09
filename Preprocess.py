#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import pickle
from sklearn import preprocessing
import time
import config

class Preprocess:
	def __init__(self, rootDir):
		self.rootDir = rootDir
		self.loan_dtypes = {'credit_score': np.int16, 'first_payment_date': np.float32, 'first_time': object, 'mat_date': np.float32, 'msa':np.float32, 'mortgage_insurance':np.float32, 'channel': object, 'PPM': object, 'product_type': object, 'state': object, 'property_type': object, 'postal_code': np.float32, 'loan_number': object, 'num_units':np.int16, 'occupancy':object, 'CLTV': np.float32, 'DTI': np.float32, 'UPB': np.float32, 'LTV': np.float32, 'original_interest_rate':np.float32, 'purpose': object, 'original_loan_term': np.float32, 'num_borrowers': np.int16, 'seller': object, 'servicer': object, 'super_conforming_flag': object, 'preharp_seq_num': object}
		# Define column names for loan data
		self.loan_name = ['credit_score', 'first_payment_date', 'first_time', 'mat_date', 'msa', 'mortgage_insurance', 'num_units', 'occupancy', 'CLTV', 'DTI', 'UPB', 'LTV', 'original_interest_rate','channel', 'PPM','product_type', 'state', 'property_type', 'postal_code', 'loan_number','purpose', 'original_loan_term', 'num_borrowers', 'seller', 'servicer', 'super_conforming_flag', 'preharp_seq_num']
		# Define data types for monthly data
		self.monthly_dtypes = {'loan_number': object, 'monthly_period': np.float32, 'current_UPB': object, 'current_status': object, 'loan_age': np.float32, 'remaining_month': object, 'repurchase_flag': object, 'modification_flag': object, 'zero_balance': np.float32, 'zero_balance_date': np.float32, 'current_interest_rate': np.float32, 'current_deferred_UPB': np.float32, 'DDLPI': np.float32, 'mi_recovery': object, 'net_sales_procedees': object, 'non_mi_recovery': object, 'expenses': np.float32, 'legal_cost': np.float32,'maintainance_cost': np.float32, 'tax': np.float32, 'misc_cost': np.float32, 'actual_loss': np.float32, 'modification_cost': object, 'step_modification_flag': object, 'deferred_payment_mod': object, 'ELTV': np.float32}
		# Define column names for monthly data
		self.monthly_name = ['loan_number', 'monthly_period', 'current_UPB', 'current_status', 'loan_age','remaining_month', 'repurchase_flag', 'modification_flag', 'zero_balance', 'zero_balance_date','current_interest_rate', 'current_deferred_UPB', 'DDLPI', 'mi_recovery','net_sales_procedees', 'non_mi_recovery', 'expenses', 'legal_cost', 'maintainance_cost','tax', 'misc_cost', 'actual_loss', 'modification_cost', 'step_modification_flag', 'deferred_payment_mod', 'ELTV']
		# Columns to one hot in load data
		self.loan_one_hot_columns = ['first_time', 'occupancy', 'channel', 'PPM', 'product_type', 'property_type', 'purpose', 'seller', 'servicer','super_conforming_flag', 'preharp_seq_num']
		# Columns to one hot in monthly data
		self.monthly_one_hot_columns = ['repurchase_flag', 'modification_flag', 'step_modification_flag', 'deferred_payment_mod']
		self.num_chunk = 100
		self.logfile = open(os.path.join('./Data/', 'info2.txt'), 'w+')
		self.logfile.close()

	def ReadData(self, dirName, filenames):
		print('dirname='+dirName+'\n')
		for f in filenames:
			if 'time' in f:
				print('file name='+f+'\n')
				monthly_data = pd.read_csv(os.path.join(dirName, f), delimiter='|', header = None, dtype=self.monthly_dtypes, names=self.monthly_name)
				monthly_data.step_modification_flag = monthly_data.step_modification_flag.astype('category', categories = config.unique_values['unique_step_modification_flags'])
				print('monthly data rows = ', monthly_data.shape)
				monthly_data_temp = pd.get_dummies(monthly_data, columns=self.monthly_one_hot_columns, sparse=True)
			else:
				print('file name='+f+'\n')
				loan_data = pd.read_csv(os.path.join(dirName, f), delimiter='|', header = None, dtype=self.loan_dtypes, names=self.loan_name)
				loan_data.super_conforming_flag = loan_data.super_conforming_flag.astype('category', categories = config.unique_values['unique_super_conforming_flags'])
				loan_data.PPM = loan_data.PPM.astype('category', categories = config.unique_values['unique_PPM'])
				loan_data.channel = loan_data.channel.astype('category', categories = config.unique_values['unique_channel'])
				loan_data.occupancy = loan_data.occupancy.astype('category', categories = config.unique_values['unique_occupancy'])
				loan_data.first_time = loan_data.first_time.astype('category', categories = config.unique_values['unique_first_time'])
				print('loan data shape = ', loan_data.shape)
				loan_data_temp = pd.get_dummies(loan_data, columns=self.loan_one_hot_columns, sparse=True)
		print(str(type(monthly_data_temp)), ' ', str(type(loan_data_temp)))
		return (loan_data_temp, monthly_data_temp)

	def PreprocessData(self):
		index = 0
		max_loan_duration = -1
		self.logfile = open(os.path.join('./Data/', 'info2.txt'), 'w+')
		for dirName, _, fileList in os.walk(self.rootDir):
			if len(fileList) > 1:
				loan_data, monthly_data = self.ReadData(dirName, fileList)
				print("After read data size: ", loan_data.shape, monthly_data.shape)
				chunk_size = int(loan_data.shape[0] / self.num_chunk)
				for i in range(0, self.num_chunk):
					curr_loans = loan_data.iloc[i*chunk_size:min((i+1)*chunk_size, loan_data.shape[0]), :]
					print('curr_loans.shape = ', curr_loans.shape)
					print(curr_loans.columns)
					print(monthly_data.columns)
					df = pd.merge(curr_loans, monthly_data, on='loan_number')
					# Following line is only for training set
					df = df.loc[(df.monthly_period/100).astype(int) < 2016, :]
					# Create Lookup
					lookup = df[['loan_number']].groupby(['loan_number']).size().reset_index(name='counts')
					lookup_table = []
					curr = 0
					for ind, r in lookup.iterrows():
						lookup_table.append((r[0], curr, curr+r[1]))
						curr = curr+r[1]
						max_loan_duration = max(max_loan_duration, r[1])
					lookup_table = pd.DataFrame(lookup_table)
					# Save as .h5
					print("df to be saved: size = ", df.shape)
					print("lookup_shape= ", lookup_table.shape)
					df.to_hdf(os.path.join('./Data/', 'Loan_Data_Train5.h5'), key='X_'+str(index), format='t', complevel=9)
					lookup_table.to_hdf(os.path.join('./Data/', 'Loan_Data_Train5.h5'), key='lookup_'+str(index), format='t', complevel=9)
					index = index + 1
					if index % 100 == 0:
						self.logfile.write(str(index) + ' chunks processed')
						self.logfile.write(str(max_loan_duration) + 'max loan duration')
		self.logfile.write(str(max_loan_duration) + 'max loan duration')
		self.logfile.close()


	def PrepareTrainTestSet(self):
		max_loan_duration = -1
		hdf_file_train = os.path.join(self.rootDir, 'Loan_Data_train.h5')
		hdf_file_train2 = os.path.join(self.rootDir, 'Loan_Data_train2.h5')
		train_index = 0

		with pd.HDFStore(hdf_file_train) as hdf:
			ln = int(len(hdf.keys())/2)

		for t in range(ln):
			df = pd.read_hdf(hdf_file_train, 'X_' + str(t))
			lookup = pd.read_hdf(hdf_file_train, 'lookup_' + str(t))

			df = df.loc[(df.monthly_period/100).astype(int) < 2016, :]
			lookup = df[['loan_number']].groupby(['loan_number']).size().reset_index(name='counts')
			lookup_table = []
			curr = 0
			for ind, r in lookup.iterrows():
				lookup_table.append((r[0], curr, curr+r[1]))
				curr = curr+r[1]
				max_loan_duration = max(max_loan_duration, r[1])
			lookup_table = pd.DataFrame(lookup_table)
			# Save as .h5
			df.to_hdf(hdf_file_train2, key='X_'+str(train_index), format='t', complevel=9)
			lookup_table.to_hdf(hdf_file_train2, key='lookup_'+str(train_index), format='t', complevel=9)
			train_index += 1

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4