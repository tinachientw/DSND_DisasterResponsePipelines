import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_path, categories_path):
    """
    Load Messages & Categories data
    
    Args:
        messages_path -> Path to the CSV file of messages
        categories_path -> Path to the CSV file of categories
    Return:
        df: dataframe contain messages and categories
    """
    
    messages = pd.read_csv(messages_path)
    categories = pd.read_csv(categories_path)
    df = pd.merge(messages, categories, on='id', how='outer')
    return df 

def clean_data(df):
    """
    Clean Categories Data
    
    Args:
        df -> data before cleaning
    Return:
        df -> data with categories cleaned up
    """
    
    # Split the categories
    categories = df['categories'].str.split(pat=';',expand=True)
    
    #Fix the categories columns name
    #row = categories.iloc[[0]]
    #category_colnames = row.transform(lambda x: x[:-2]).tolist()
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    
    categories = (categories > 0).astype(int)
    df = df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df,categories], join='inner', axis=1)
    #df = df.drop_duplicates().sum()
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """
    Save Data to SQLite Database Function
    
    Args:
        df -> data with categories cleaned up
        database_file -> Path to SQLite destination database
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')

def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:] 

        print('Loading data...')
        
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...')
        save_data(df, database_filepath)
        
        print('Cleaned data has been saved to database!')
    
    else: # Print the help message so that user can execute the script with correct parameters
        print("Please provide the arguments correctly: \n\
			1) Path to the messages CSV file\n\
			2) Path to the categories CSV file\n\
			3) Path to SQLite database \n\
			Ex:python process_data.py disaster_messages.csv disaster_messages.csv disaster.db ")

if __name__ == '__main__':
    main()