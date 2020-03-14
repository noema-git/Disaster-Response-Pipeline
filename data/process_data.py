# PREPROCESSING THE DATA
# Disaster Response Pipeline Project
# 
# The script takes the file paths of the two datasets and database, 
# cleans the datasets, and stores the clean data into a SQLite database 
# in the specified database file path. # It merges the messages and 
# categories datasets, splits the categories column into separate, 
# clearly named columns, converts values to binary, and drops duplicates.
#
# Arguments:
#     1) CSV file containing messages (ipnut)
#     2) CSV file containing categories (input)
#     3) SQL destination database (output)
#

# Load required libraries
import sys
import pandas as pd
import sqlite3 as sqlite

def load_data(messages_filepath, categories_filepath):
    """
    Load Data function   
    - Takes as inputs two CSV files
    - Merges them into a single dataframe
    - Outputs a Pandas DataFrame.

    Arguments:
        messages_filepath: the path to messages csv file
        categories_filepath: the path to categories csv file
    Output:
        df: Merged data in Pandas DataFrame
    """

    # Load the input files
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)

    # Merge the input files in one DataFrame
    df = pd.merge(messages,categories,on='id')
    return df 


def clean_data(df):
    """
    Clean data function

    Arguments:
        df: Pandas DataFrame (raw information)
    Output:
        df: Pandas DataFrame (cleaned information)
    """

    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)

    # Select the first row of the categories 
    firstrow = categories.iloc[0,:]

    # Extract new column names for categories
    category_colnames = firstrow.apply(lambda x:x[:-2])

    # Rename the columns
    categories.columns = category_colnames
    
    # Get the last character of the string and convert it numeric
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])

    # Drop the original column 'categories' from the DataFrame
    df.drop('categories', axis=1, inplace=True)

    # Concat the DataFrames
    df = pd.concat([df, categories], axis=1)

    # Drop all duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save Data function
    
    Arguments:
        df: Clean data Pandas DataFrame
        database_filename: Database file and destination path (.db)
    Output:
        None
    """

    # Connect to SQLite database
    sql_con = sqlite.connect(database_filename)

    # Save Datafame to SQL Database
    df.to_sql(name='DRP', con=sql_con, if_exists="replace")

    sql_con.close()

def main():
    """
    Main function
    
    The function implements the ETL pipeline
        1) Load Data from .csv
        2) Clean Data and pre-processing
        3) Save Data to SQL database
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()