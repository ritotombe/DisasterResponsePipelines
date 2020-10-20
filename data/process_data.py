import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Function to load input data (in csv) to a pandas dataframe
    Input
    - messages_filepath: path to the message dataset
    - categories_filepath: path to the categories dataset
    Output
    - df: a dataframe with message and categories datasets merged
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id", how="inner")
    return df


def clean_data(df):
    """ Clean and format data to prepare to be analysed
    Input
    - df: dataframe to be cleaned
    Output
    - df: cleaned data frame
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype("int")
        
    df.drop(columns=["categories"], inplace=True)
    df = pd.concat([df,categories], axis=1)
    df = df.drop_duplicates()
    
    # remove value 2
    df.loc[df.related == 2, 'related']=1
    return df


def save_data(df, database_filename):
    """ Save dataframe as a SQLite database
    Input
    - df: dataframe to save
    - database_filename: name of the database
    Output
    None - a SQLite should be produced
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql("MESSAGE_CATEGORIES", engine, index=False)
    pass  


def main():
    if len(sys.argv) == 4:
        print(pd.show_versions())
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