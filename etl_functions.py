import pandas as pd
import os
import configparser
import datetime as dt
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg
from pyspark.sql import SQLContext
from pyspark.sql.functions import count, col, udf, dayofmonth, dayofweek, month, year, weekofyear
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import *

## functions used on etl
def create_fact_table_immigration(spark, df, output_data):
    """Create a country dimension table from the global temperatures and immigration data.
    :param spark: spark session
    :param df: dataframe of immigration events
    :param visa_type_df: dataframe of global land temperatures data.
    :param output_data: path to where to write dimension df to
    :return: dataframe representing calendar dimension
    """
    
    dimension_df = get_visa_type_dimension(spark, output_data) # get visa_type dimension
    dimension_df.createOrReplaceTempView("visa_view") # create a view for visa type dimension
    get_datetime = udf(lambda x: (dt.datetime(1960, 1, 1).date() + dt.timedelta(x)).isoformat() if x else None) # converting arrival date in SAS format to datetime object with udf
    ndf = ndf.withColumnRenamed('ccid', 'record_id')# rename columns 
    ndf = ndf.withColumnRenamed('i94res', 'country_residence_code')# rename columns
    ndf = ndf.withColumnRenamed('i94addr', 'state_code') # rename columns
    ndf.createOrReplaceTempView("immigration_view")# create a view for immigration
    ndf = spark.sql(
        """
        SELECT immigration_view.*,visa_view.visa_type_key
        FROM immigration_view
        LEFT JOIN visa_view ON visa_view.visatype=immigration_view.visatype
        """)
    ndf = ndf.withColumn("arrdate", get_datetime(df.arrdate))# convert arrival date into datetime object
    ndf = ndf.drop(ndf.visatype)# drop visatype key
    ndf.write.parquet(output_data + "immigration_fact", mode="overwrite")# write dimension to parquet file
    return ndf

############################################################################################################

def create_dimension_table_demographics(df, output_data):
    """US demographics dimension table creation
    :param df: dataframe of us demographics df data 
    :param output_data: path towhere to write dimension dataframe to
    :return: dataframe representing demographics dimension
    """
    dim_df = df.withColumnRenamed('Median Age', 'median_age') 
    dim_df = dim_df.withColumnRenamed('Male Population', 'male_population') 
    dim_df = dim_df.withColumnRenamed('Female Population', 'female_population') 
    dim_df = dim_df.withColumnRenamed('Total Population', 'total_population') 
    dim_df = dim_df.withColumnRenamed('Number of Veterans', 'number_of_veterans') 
    dim_df = dim_df.withColumnRenamed('Foreign-born', 'foreign_born')
    dim_df = dim_df.withColumnRenamed('Average Household Size', 'average_household_size') 
    dim_df = dim_df.withColumnRenamed('State Code', 'state_code')
    dim_df = dim_df.withColumn('id', monotonically_increasing_id()) # lets add an id column 1 by 1 automaticaly
    dim_df.write.parquet(output_data + "demographics", mode="overwrite") # write to parquet file dimmension table
    return dim_df

############################################################################################################

def create_dimension_table_visa_type(df, output_data):
    """Dimension table for Visa type creation from the immigration data.
    :param df: dataframe of immigration info
    :param output_data: path to where write dimension dataframe
    :return: dataframe representing
    """
    visatype_df = df.select(['visatype']).distinct()# create visatype df from visatype column
    visatype_df = visatype_df.withColumn('visa_type_key', monotonically_increasing_id())# add an id column
    visatype_df.write.parquet(output_data + "visatype", mode="overwrite")# write dimension to parquet file
    return visatype_df

############################################################################################################

def get_visa_type_dimension(spark, output_data):
    return spark.read.parquet(output_data + "visatype")

############################################################################################################

def create_dimension_table_country(spark, df, temp_df, output_data, mapping_file):
    """Creates a country dimension table
    :param spark: session object
    :param df: dataframe of immigration events
    :temp_df: dataframe of global temperatures data.
    :param output_data: path to where to write dimension dataframe to
    :param mapping_file: csv file which maps on one file country codes to country names 
    :return: result dataframe representing calendar dimension
    """
    df.createOrReplaceTempView("immigration_view")# creating temporary view for immigration
    mapping_file.createOrReplaceTempView("country_codes_view")# creating temporary view for countries codes
    aggregate_temp = aggregate_temperature(temp_df)# getting the aggregated temperature
    aggregate_temp.createOrReplaceTempView("average_temperature_view")# create temporary view for countries average temps data
    country_df = spark.sql(
        """
        SELECT i94res as country_code, Name as country_name
        FROM immigration_view
        LEFT JOIN country_codes_view
        ON immigration_view.i94res=country_codes_view.code
        """
    ).distinct()# create country dimension using SQL
    country_df.createOrReplaceTempView("country_view")# create temp country view
    country_df = spark.sql(
        """
        SELECT country_code, country_name, average_temperature
        FROM country_view
        LEFT JOIN average_temperature_view
        ON country_view.country_name=average_temperature_view.Country
        """
    ).distinct()
    country_df.write.parquet(output_data + "country", mode="overwrite")# write the dimension to a parquet file
    return country_df

############################################################################################################

def create_immigration_calendar_dimension(df, output_data):
    """This function to create an immigration calendar dimension table using the arrival date column
    :param df: dataframe of immigration events
    :param output_data: where to write dimension dataframe
    :return: dataframe representing calendar dimension
    """
    get_datetime = udf(lambda x: (dt.datetime(1960, 1, 1).date() + dt.timedelta(x)).isoformat() if x else None)# create a udf to convert arrival date in SAS format to datetime object
    cal_df = df.select(['arrdate']).withColumn("arrdate", get_datetime(df.arrdate)).distinct() # create initial calendar df from arrdate column
    cal_df = cal_df.withColumn('arrival_day', dayofmonth('arrdate'))
    cal_df = cal_df.withColumn('arrival_week', weekofyear('arrdate'))
    cal_df = cal_df.withColumn('arrival_month', month('arrdate'))
    cal_df = cal_df.withColumn('arrival_year', year('arrdate'))
    cal_df = cal_df.withColumn('arrival_weekday', dayofweek('arrdate'))# expand df by adding other calendar columns
    cal_df = cal_df.withColumn('id', monotonically_increasing_id())# create an id field in calendar df
    partition_columns = ['arrival_year', 'arrival_month', 'arrival_week']
    cal_df.write.parquet(output_data + "immigration_calendar", partitionBy=partition_columns, mode="overwrite")# write the calendar dimension to parquet file
    return cal_df

############################################################################################################

# Perform quality checks here
def quality_checks(df, table_name):
    """Quality checks.:
    :param df:  dataframe that will be checked
    :param table_name: name of table that will be checked
    """
    total = df.count() # total number of rows on the table

    if total == 0: # if the result is eq to 0, the table does not pass
        print(f"Table NOK! Data quality check failed for {table_name} with zero records on it!")
    else: #if not, table pass on the test
        print(f"Table OK! Data quality check passed for {table_name} with {total:,} records on it.")
    return 0