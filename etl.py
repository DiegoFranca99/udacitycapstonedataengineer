import os
import pandas as pd
import configparser
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.functions import from_unixtime, to_timestamp
from pyspark.sql.functions import col, hour, dayofmonth, dayofweek, month, year, weekofyear
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import monotonically_increasing_id
import utility
import etl_functions

config = configparser.ConfigParser()
config.read('config.cfg')
os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "saurfang:spark-sas7bdat:2.0.0-s_2.11") \
        .enableHiveSupport() \
        .getOrCreate()
    return spark

############################################################################################################

def process_immigration_data(spark, input_data, output_data, file_name, temperature_file, mapping_file):
    """Process for creating fact and dimension tables on the  immigration df 
    Parameters:
    spark (SparkSession): spark session instance
    input_data (string): input file path
    output_data (string): output file path
    file_name (string): immigration input file name
    mapping_file (pandas dataframe): dataframe which maps country codes to country names instead
    temperature_file (string): global_temperatures input file name
    """
    immigration_file = input_data + file_name # get the file path to the immigration data
    immigration_df = spark.read.format('com.github.saurfang.sas.spark').load(immigration_file)# read immigration data file
    immigration_df = utility.clean_spark_immigration_data(immigration_df)# clean immigration spark dataframe
    visatype_df = etl_functions.create_dimension_table_visa_type(immigration_df, output_data)# create visa_type dimension table
    calendar_df = etl_functions.create_immigration_calendar_dimension(immigration_df, output_data)# create calendar dimension table
    temp_df = process_global_land_temperatures(spark, input_data, temperature_file)# get global temperatures data
    dim_df = etl_functions.create_dimension_table_country(spark, immigration_df, temp_df, output_data, mapping_file)# create country dimension table
    fact_df = etl_functions.create_fact_table_immigration(spark, immigration_df, output_data)# create immigration fact table

############################################################################################################

def process_demographics_data(spark, input_data, output_data, file_name):
    """Process the demographics data and create the usa_demographics_dim table to be used
    Parameters:
    spark (SparkSession): spark session instance
    input_data (string): input file path
    output_data (string): output file path
    file_name (string): usa_demographics csv file name with the data
    """
    file = input_data + file_name# load demographics data
    demographics_df = spark.read.csv(file, inferSchema=True, header=True, sep=';')
    new_demographics_df = utility.clean_spark_demographics_data(demographics_df)# clean demographics data
    df = etl_functions.create_dimension_table_demographics(new_demographics_df, output_data)# create demographic dimension table

############################################################################################################

def process_global_land_temperatures(spark, input_data, file_name):
    """Process the global land temperatures data and return a dataframe to be used
    Parameters:
    spark (SparkSession): spark session instance
    input_data (string): input file path
    output_data (string): output file path
    file_name (string):  global land temperatures file name with the data
    """
    file = input_data + file_name # load data
    temperature_df = spark.read.csv(file, header=True, inferSchema=True)
    new_temperature_df = utility.clean_spark_temperature_data(temperature_df)# clean the temperature data
    return new_temperature_df

############################################################################################################

def main():
    spark = create_spark_session()
    input_data = "s3a://capstone-bucket-df/"
    output_data = "s3a://capstone-bucket-df/"

    immigration_file_name = 'i94_apr16_sub.sas7bdat'
    temperature_file_name = 'GlobalLandTemperaturesByCity.csv'
    usa_demographics_file_name = 'us-cities-demographics.csv'

    mapping_file = input_data + "i94res.csv"
    mapping_file = spark.read.csv(mapping_file, header=True, inferSchema=True)# load the i94res to country mapping data
    process_immigration_data(spark, input_data, output_data, immigration_file_name, temperature_file_name, mapping_file)
    process_demographics_data(spark, input_data, output_data, usa_demographics_file_name)

if __name__ == "__main__":
    main()