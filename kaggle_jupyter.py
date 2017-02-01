# Import libraries
import pandas as pd
import numpy as np
import os
import pprint
import matplotlib.pyplot as plt
import autotime

# Load autotime to get cell execution time automatically
%load_ext autotime

# Inline plots
%matplotlib inline

# Setup environment
os.chdir('/local/home/jsahewal/machineLearning')

# Store input filenames, output filenames and date columns
train_in_file = 'data/train.csv'
test_in_file = 'data/test.csv'
dest_in_file = 'data/destinations.csv'
out_file = 'data/dataFrames.h5'
date_columns = ['date_time', 'srch_ci', 'srch_co']

# Load data
if os.path.exists(out_file):
    # Initialize HDFStore
    store = pd.HDFStore(out_file)
    # Load the data from h5
    for key in store.keys():
        var_name = key.split("/")[1]
        vars()[var_name] = store[var_name]
else:
    # Load data from csv files
    train_df = pd.read_csv(train_in_file, index_col='srch_destination_id', parse_dates=date_columns)
    test_df = pd.read_csv(test_in_file, index_col='srch_destination_id', parse_dates=date_columns)
    dest_df = pd.read_csv(dest_in_file, index_col='srch_destination_id')
    # Initialize HDFStore
    store = pd.HDFStore(out_file)
    # Save data to h5 store
    store['train_df'] = train_df
    store['test_df'] = test_df
    store['dest_df'] = dest_df
