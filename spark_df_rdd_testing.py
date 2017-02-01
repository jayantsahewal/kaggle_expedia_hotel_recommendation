### Extra commands
# conda update --all
# import rlcompleter, readline
# pd.show_versions()
# train_file= 'train.csv'
# train_df = pd.read_csv(train_file)
# pprint.pprint(test_df.describe().to_string())
# pd.set_option('display.width', 240)
# readline.parse_and_bind('tab:complete')
# w = %who_ls int str DataFrame list dict
# ssh -N -f -L localhost:8889:localhost:8890 <USER>@<REMOTE_HOST>
# jupyter notebook --no-browser --port=8890


### Useful links
# http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#
# http://spark.apache.org/docs/latest/sql-programming-guide.html


### Import libraries
import pandas as pd
import numpy as np
import os
import pprint
import matplotlib.pyplot as plt
import autotime
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

### Load autotime to get cell execution time automatically
%load_ext autotime

### Inline plots
%matplotlib inline

### Setup environment
os.chdir('/local/home/jsahewal/machineLearning')

### Store input filenames, output filenames and date columns
train_in_file = 'data/train.csv'
test_in_file = 'data/test.csv'
dest_in_file = 'data/destinations.csv'
hdf5_out_file = 'data/dataFrames.h5'
date_columns = ['date_time', 'srch_ci', 'srch_co']

### Load data
if os.path.exists(hdf5_out_file):
    ### Initialize HDFStore
    store = pd.HDFStore(hdf5_out_file)
    ### Load the data from h5
    for key in store.keys():
        var_name = key.split("/")[1]
        vars()[var_name] = store[var_name]
else:
    ### Load data from csv files
    train_df = pd.read_csv(train_in_file, index_col='srch_destination_id', 
							parse_dates=date_columns)
    test_df = pd.read_csv(test_in_file, index_col='srch_destination_id', 
							parse_dates=date_columns)
    dest_df = pd.read_csv(dest_in_file, index_col='srch_destination_id')
    ### Initialize HDFStore
    store = pd.HDFStore(hdf5_out_file)
    ### Save data to h5 store
    store['train_df'] = train_df
    store['test_df'] = test_df
    store['dest_df'] = dest_df

### Create Spark Session
spark = SparkSession.builder.appName("KaggleExpedia") \
		.config("spark.some.config.option", "some-value").getOrCreate()

### Experimentation with Spark RDDs

### Load data in Spark RDD
test_spark_rdd = spark.read.format("csv").option("header", 
													"true").csv(test_in_file)
train_spark_rdd = spark.read.format("csv").option("header", 
													"true").csv(train_in_file)

### Create views on top of Spark RDDs
train_spark_rdd.createOrReplaceTempView("train_spark_rdd_table")
test_spark_rdd.createOrReplaceTempView("test_spark_rdd_table")

### Create Spark SQL queries
sqlDF1 = spark.sql("SELECT distinct site_name FROM train_spark_rdd_table")
sqlDF2 = spark.sql("SELECT site_name, count(distinct hotel_cluster) FROM \
						train_spark_rdd_table GROUP BY site_name")

### Execute Spark SQL queries
sqlDF1.show()
sqlDF2.show()

### Experimentation with Spark DataFrames
### DO NOT RUN THE FOLLOWING TIME CONSUMING COMMANDS
### Load data in Spark DataFrame. 
# test_spark_df = spark.createDataFrame(test_df)
# train_spark_df = spark.createDataFrame(train_df)

### Create views on top of Spark DataFrames
# train_spark_df.createOrReplaceTempView("train_spark_df_table")
# test_spark_df.createOrReplaceTempView("test_spark_df_table")

### Create Spark SQL queries
# sqlDF3 = spark.sql("SELECT distinct site_name FROM train_spark_df_table")
# sqlDF4 = spark.sql("SELECT site_name, count(distinct hotel_cluster) FROM \
# 						train_spark_df_table GROUP BY site_name")

### Execute Spark SQL queries
# sqlDF3.show()
# sqlDF4.show()

# Sample Spark DataFrame commands
# test_spark_df.count()
# test_spark_df.cube("site_name", "posa_continent").count() \
			# .orderBy("site_name", "posa_continent").show()
# test_spark_df.cube("site_name", "posa_continent").count() \
			# .orderBy("site_name", "posa_continent").collect()
# test_spark_df.describe(['site_name']).show()
# test_spark_df.describe(['site_name']).show()
