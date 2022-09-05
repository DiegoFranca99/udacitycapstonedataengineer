# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import configparser
import datetime as dt

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg
from pyspark.sql import SQLContext
from pyspark.sql.functions import isnan, when, count, col, udf, dayofmonth, dayofweek, month, year, weekofyear
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import *

import plotly.plotly as py
import plotly.graph_objs as go
import requests
requests.packages.urllib3.disable_warnings()

############################################################################################################

def visualize_missing_values_spark(df):
    """Plot missing values
    :param df: spark dataframe
    """
    nan_count_df = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas() # create a df with missing values count per column
    nan_count_df = pd.melt(nan_count_df, var_name='cols', value_name='values') # convert dataframe from wide format to long format
    total = df.count() # count total records in df
    nan_count_df['% missing values'] = 100*nan_count_df['values']/total # adding (%) of missing values column
    a = nan_count_df['% missing values'].max()
    
    plt.rcdefaults()
    plt.figure(figsize=(10,5))
    ax = sns.barplot(x="cols", y="% missing values", data=nan_count_df)
    if a + 5 < 100:
        ax.set_ylim(0, a)
    else:
        ax.set_ylim(0, 100)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    plt.show()
    
############################################################################################################

def clean_immigration_data_spark(df):
    """Clean immigration dataframe
    :param df: dataframe with monthly updated immigration dataset
    :return: clean dataframe after process run 
    """
    total = df.count()
    print(f'Total records in dataframe priviously in the dataframe: {total:,}')
    
    dropping_columns = ['occup', 'entdepu','insnum'] # columns shown on the plot on jupyter
    df = df.drop(*dropping_columns)# dropping those columns
    df = df.dropna(how='all')# drop rows that have all elements missing on this dataframe
    new_total = df.count()
    print(f'After dropping those columns and the rows with missing values, the total number rows are: {new_total:,}')
    return df

############################################################################################################

def create_immigration_calendar_dimension(df, output_data):
    """Creates an immigration calendar dimension table.:
    :param df: dataframe of immigration provided
    :param output_data: path to write dimension dataframe in 
    :return: spark dataframe calendar set
    """
    get_datetime = udf(lambda x: (dt.datetime(1960, 1, 1).date() + dt.timedelta(x)).isoformat() if x else None) # converting arrival date in SAS format to datetime object with udf
    cal_df = df.select(['arrdate']).withColumn("arrdate", get_datetime(df.arrdate)).distinct()# create initial calendar df from arrdate column
    cal_df = cal_df.withColumn('arrival_day', dayofmonth('arrdate'))
    cal_df = cal_df.withColumn('arrival_week', weekofyear('arrdate'))
    cal_df = cal_df.withColumn('arrival_month', month('arrdate'))
    cal_df = cal_df.withColumn('arrival_year', year('arrdate'))
    cal_df = cal_df.withColumn('arrival_weekday', dayofweek('arrdate'))# expand df by adding other calendar columns
    cal_df = cal_df.withColumn('id', monotonically_increasing_id())# create an id field in calendar df
    partition_columns = ['arrival_year', 'arrival_month', 'arrival_week']# write the calendar dimension to parquet file
    cal_df.write.parquet(output_data + "immigration_calendar", partitionBy=partition_columns, mode="overwrite")
    return cal_df

############################################################################################################

def clean_temperature_data_spark(df):
    """Clean df global temperature
    :param df: dataframe representing global temperatures
    :return: clean dataframe after process run
    """
    total = df.count()
    print(f'Total number of rows on the dataframe before cleaning process: {total:,}') #printing the amount
    df = df.dropna(subset=['AverageTemperature'])# dropping rows that have missing values on average temperature
    total_after_missing = df.count()
    print('Total number of rows after dropping missing values: {:,}'.format(total-total_after_missing))#printing the amount after calculation
    df = df.drop_duplicates(subset=['dt', 'City', 'Country'])# drop duplicate rows
    total_after_duplicate = df.count()
    print('Total number of rows after dropping duplicate values: {:,}'.format(total_after_missing-total_after_duplicate))#printing the amount after calculation
    return df

############################################################################################################

def aggregate_temperature(df):
    """Aggregate fluction clean temperature data at country level of the dataframe inputed 
    :param df: spark dataframe of clean global temperaturs data
    :return:dataframe consisting of countries average temperatures
    """
    new_aggregate_df = df.select(['Country', 'AverageTemperature']).groupby('Country').avg() # group df by country values and calculate avg of temperature
    new_aggregate_df = new_aggregate_df.withColumnRenamed('avg(AverageTemperature)', 'average_temperature') #Renamed the columns and aply on the new df
    return new_aggregate_df

############################################################################################################

def clean_demographics_data_spark(df):
    """Clean the US demographics dataset according to the duplicates values on the dataframe
    :param df: dataframe of USdemographics dataset
    :return: cleaned dataframe
    """
    subset_cols = ['Male Population','Female Population','Number of Veterans','Foreign-born','Average Household Size']# columns with missing values
    new_df = df.dropna(subset=subset_cols) #dropping those columns
    total = df.count()
    rows_dropped = total-new_df.count()  #calculate the total of rows dropped
    print("Number of rows dropped that had missing values: {}".format(rows_dropped)) # printing
    new_df2 = new_df.dropDuplicates(subset=['City', 'State', 'State Code', 'Race']) # dropping columns duplicate 
    rows_dropped_duplicates = new_df.count()-new_df2.count()
    print(f"Rows dropped after accounting for duplicates on the dataframe: {rows_dropped_duplicates}")
    last_total = new_df2.count()
    print(f"Total rows after running the process of cleaning on the df: {last_total}")
    return new_df2

############################################################################################################
############################################################################################################
############################################################################################################

def print_formatted_float(number):
    print('{:,}'.format(number))
    