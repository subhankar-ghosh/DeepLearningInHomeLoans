import numpy as np
import os
import pandas as pd
import pickle
from sklearn import preprocessing
import time

class Preprocess:
	def __init__(self, rootDir):
		```
		# Constructor of the Preprocess class.
		# Defines datatypes for all columns of the data.
		# Defines load names or class variable.
		# Defines variables to be one hot encoded.
		# Defines chunk size - since the files are too huge they need to be read in chunks.
		# Args: 
		#  	rootDir: Path to the directory that contains raw data
		
		```
		self.rootDir = rootDir
		self.loan_dtypes = {'credit_score': np.int64, 'first_payment_date': np.float32, 'first_time': object, 'mat_date': np.float32, 'msa':np.float64,\
		 'mortgage_insurance':np.float64, 'channel': object, 'PPM': object, 'product_type': object, 'state': object, 'property_type': object, \
		 'postal_code': np.float64, 'loan_number': object, 'num_units':np.int64, 'occupancy':object, 'CLTV': np.float64, 'DTI': np.float64,\
		  'UPB': np.float64, 'LTV': np.float64, 'original_interest_rate':np.float64, 'purpose': object, 'original_loan_term': np.float64, \
		  'num_borrowers': np.int64, 'seller': object, 'servicer': object, 'super_conforming_flag': object, 'preharp_seq_num': object}
		# Define column names for loan data
		self.loan_name = ['credit_score', 'first_payment_date', 'first_time', 'mat_date', 'msa', 'mortgage_insurance', 'num_units', 'occupancy', 'CLTV', 'DTI', 'UPB', 'LTV', 'original_interest_rate','channel', 'PPM','product_type', 'state', 'property_type', 'postal_code', 'loan_number','purpose', 'original_loan_term', 'num_borrowers', 'seller', 'servicer', 'super_conforming_flag', 'preharp_seq_num']
		# Define data types for monthly data
		self.monthly_dtypes = {'loan_number': object, 'monthly_period': np.float32, 'current_UPB': object, 'current_status': object,\
		 'loan_age': np.float32, 'remaining_month': object, 'repurchase_flag': object, 'modification_flag': object, 'zero_balance': np.float32,\
		  'zero_balance_date': np.float32, 'current_interest_rate': np.float32, 'current_deferred_UPB': np.float32, 'DDLPI': np.float32,\
		   'mi_recovery': object, 'net_sales_procedees': object, 'non_mi_recovery': object, 'expenses': np.float32, 'legal_cost': np.float32,\
		   'maintainance_cost': np.float32, 'tax': np.float32, 'misc_cost': np.float32, 'actual_loss': np.float32, 'modification_cost': object,\
		    'step_modification_flag': object, 'deferred_payment_mod': object, 'ELTV': np.float32}
		# Define column names for monthly data
		self.monthly_name = ['loan_number', 'monthly_period', 'current_UPB', 'current_status', 'loan_age','remaining_month', 'repurchase_flag',\
		 'modification_flag', 'zero_balance', 'zero_balance_date','current_interest_rate', 'current_deferred_UPB', 'DDLPI', 'mi_recovery',\
		 'net_sales_procedees', 'non_mi_recovery', 'expenses', 'legal_cost', 'maintainance_cost','tax', 'misc_cost', 'actual_loss', 'modification_cost',\
		  'step_modification_flag', 'deferred_payment_mod', 'ELTV']
		# Columns to one hot in load data
		self.loan_one_hot_columns = ['first_time', 'occupancy', 'channel', 'PPM', 'product_type', 'property_type', 'purpose', 'seller', 'servicer','super_conforming_flag', 'preharp_seq_num']
		# Columns to one hot in monthly data
		self.monthly_one_hot_columns = ['repurchase_flag', 'modification_flag', 'step_modification_flag', 'deferred_payment_mod']
		self.num_chunk = 100
		self.logfile = open(os.path.join(self.rootDir, 'info.txt'), 'w')
		self.logfile.close()

	def ReadData(self, dirName, filenames):
		```
		# Reads data from raw files. Two types of files are there.
		# (1) Loan data files - files with information about every single loan
		# (2) Monthly data files - Files with each row corresponding to one month data of a loan,
		# 	Every monthly data row has a loan number which corresponds to a unique loan in
		# 	Loan data file.
		```
		loan_data_temp, monthly_data_temp = None, None
		for f in filenames:
			if 'time' in f:
				print(f)
				monthly_data = pd.read_csv(os.path.join(dirName, f), delimiter='|', header = None, dtype=self.monthly_dtypes, names=self.monthly_name)
				print('Read')
				monthly_data_temp = pd.get_dummies(monthly_data, columns=self.monthly_one_hot_columns)
				print(f + ' processed')
			else:
				print(f)
				loan_data = pd.read_csv(os.path.join(dirName, f), delimiter='|', header = None, dtype=self.loan_dtypes, names=self.loan_name)
				loan_data_temp = pd.get_dummies(loan_data, columns=self.loan_one_hot_columns)
				print(f + ' processed')
		return (loan_data_temp, monthly_data_temp)

	def PreprocessData(self):
		'''
		Preprocesses every file and populate lookup table.
		Lookup table can be thought of as a hash for easy retrieval during training.
			Lookup table stores loan numbers as keys and corresponding monthly data
			rows as values.
		'''
		index = 0
		max_loan_duration = -1
		self.logfile = open(os.path.join('./', 'info.txt'), 'w+')
		for dirName, _, fileList in os.walk(self.rootDir):
			if len(fileList) > 1:
				loan_data, monthly_data = self.ReadData(dirName, fileList)
				chunk_size = int(loan_data.shape[0] / self.num_chunk)
				for i in range(0, self.num_chunk):
					curr_loans = loan_data.iloc[i*chunk_size:min((i+1)*chunk_size, loan_data.shape[0]), :]
					df = pd.merge(curr_loans, monthly_data, on='loan_number')
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
					df.to_hdf(os.path.join(self.rootDir, 'Loan_Data_16.h5'), key='X_'+str(index), format='t', complevel=9)
					lookup_table.to_hdf(os.path.join(self.rootDir, 'Loan_Data_16.h5'), key='lookup_'+str(index), format='t', complevel=9)
					index = index + 1
					if index % 100 == 0:
						self.logfile.write(str(index) + ' chunks processed')
						self.logfile.write(str(max_loan_duration) + 'max loan duration')
		self.logfile.write(str(max_loan_duration) + 'max loan duration')
		self.logfile.close()


	def PrepareTrainTestset(self):
		max_loan_duration = -1
		data_store2 = os.path.join(self.rootDir, 'Loan_Data.h5')
		data_store1 = os.path.join(self.rootDir, 'Loan_Data_3.h5')
		hdf_file_test = os.path.join(self.rootDir, 'Loan_Data_test.h5')
		hdf_file_train = os.path.join(self.rootDir, 'Loan_Data_train.h5')
		train_index = 0
		test_index = 0

		with pd.HDFStore(data_store1) as hdf:
			ln = int(len(hdf.keys())/2)

		for t in range(ln):
			df = pd.read_hdf(data_store1, 'X_' + str(t))
			lookup = pd.read_hdf(data_store1, 'lookup_' + str(t))

			df16 = df.loc[(df.first_payment_date/100).astype(int) == 2016, :]
			if df16.shape[0] > 0:
				df16.to_hdf(hdf_file_test, key='X_'+str(test_index), format='t', complevel=9)
				lookup.to_hdf(hdf_file_test, key='lookup_'+str(test_index), format='t', complevel=9)
				test_index += 1
				continue
			df17 = df.loc[(df.first_payment_date/100).astype(int) == 2017, :]
			if df17.shape[0] > 0:
				df17.to_hdf(hdf_file_test, key='X_'+str(test_index), format='t', complevel=9)
				lookup.to_hdf(hdf_file_test, key='lookup_'+str(test_index), format='t', complevel=9)
				test_index += 1
				continue

			df = df.loc[(df.first_payment_date/100).astype(int) < 2016, :]
			lookup = df[['loan_number']].groupby(['loan_number']).size().reset_index(name='counts')
			lookup_table = []
			curr = 0
			for ind, r in lookup.iterrows():
				lookup_table.append((r[0], curr, curr+r[1]))
				curr = curr+r[1]
				max_loan_duration = max(max_loan_duration, r[1])
			lookup_table = pd.DataFrame(lookup_table)
			# Save as .h5
			df.to_hdf(hdf_file_train, key='X_'+str(train_index), format='t', complevel=9)
			lookup_table.to_hdf(hdf_file_train, key='lookup_'+str(train_index), format='t', complevel=9)
			train_index += 1

		with pd.HDFStore(data_store2) as hdf:
			ln = int(len(hdf.keys())/2)

		for t in range(ln):
			df = pd.read_hdf(data_store2, 'X_' + str(t))
			lookup = pd.read_hdf(data_store2, 'lookup_' + str(t))

			df = df.loc[(df.first_payment_date/100).astype(int) < 2016, :]
			lookup = df[['loan_number']].groupby(['loan_number']).size().reset_index(name='counts')
			lookup_table = []
			curr = 0
			for ind, r in lookup.iterrows():
				lookup_table.append((r[0], curr, curr+r[1]))
				curr = curr+r[1]
				max_loan_duration = max(max_loan_duration, r[1])
			lookup_table = pd.DataFrame(lookup_table)
			# Save as .h5
			df.to_hdf(hdf_file_train, key='X_'+str(train_index), format='t', complevel=9)
			lookup_table.to_hdf(hdf_file_train, key='lookup_'+str(train_index), format='t', complevel=9)
			train_index += 1
