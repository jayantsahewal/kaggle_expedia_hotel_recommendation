# conda update --all
# pd.show_versions()
# train_file= 'train.csv'
# train_df = pd.read_csv(train_file)
# pprint.pprint(test_df.describe().to_string())

# Import libraries
import pandas
import pandas as pd
import numpy as np
import os
import pprint
import matplotlib.pyplot as plt
import rlcompleter, readline

# Setup environment
os.chdir('/local/home/jsahewal/machineLearning')
pd.set_option('display.width', 240)
readline.parse_and_bind('tab:complete')

# Store input filenames
train_in_file = 'train.csv'
test_in_file = 'test.csv'
dest_in_file = 'destinations.csv'

# Store output filenames
train_out_file = 'train.pkl'
dest_out_file = 'destinations.csv'
test_out_file = 'test.pkl'

# Load csv files
train_df = pd.read_csv(train_file, index_col='srch_destination_id')
dest_df = pd.read_csv(dest_file, index_col='srch_destination_id')
test_df = pd.read_csv(test_file, index_col='srch_destination_id')

# Save the data to .pkl files for re-usability
train_df.to_pickle(train_out_file)
dest_df.to_pickle(dest_out_file)
test_df.to_pickle(test_out_file)

# Get Unique destination with and without latent features
unique_dest_total = set(train_df.index).union(dest_df.index)
unique_dest_with_latent_features = set(dest_df.index)
unique_dest_without_latent_features = set(unique_dest.difference(unique_dest_with_latent_features))
print(len(unique_dest_total))
print(len(unique_dest_with_latent_features))
print(len(unique_dest_without_latent_features))


