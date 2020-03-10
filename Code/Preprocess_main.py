import numpy as np
import os
import pandas as pd
import pickle
from sklearn import preprocessing
from Preprocess import Preprocess
import time


def main():
    '''
    Main function to preprocess data. This function uses the Preprocess class.
    '''
    dataDirectory = './Data/ToProcessData'
    preprocessor = Preprocess(dataDirectory)
    print('Object created')
    st = time.time()
    preprocessor.PreprocessData()
    # preprocessor.PrepareTrainTestSet()
    print('Preprocess data execution ended')
    en = time.time()
    print('Time taken to process data = ', en - st, ' sec')
  
if __name__== "__main__":
    main()
