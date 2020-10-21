import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    """ Clean and format data to prepare to be analysed
    Input
    - database_filepath: path to the database
    Return
    - X: feature columns to classify
    - Y: class columns
    - category_names: a list of column names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('MESSAGE_CATEGORIES', engine)
    X = df['message'].values
    Y = df.drop(columns=['id', 'message', 'original', 'genre']).values
    category_names = df.drop(columns=['id', 'message', 'original', 'genre']).columns.tolist()
    return X, Y, category_names


def tokenize(text):
    """ Prepare text input to be processed further in the feature engineering
    Input
    - text: text to be tokenized
    Return
    - clean_tokens: tokens of the text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Prepare the model pipeling
    Input
    - None
    Return
    - cv: a grid search as an estimator/model to find best pipeline parameters
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50],
#         'clf__estimator__min_samples_split': [2],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ Run the trained model on a test set and evaluate the result by looking at accuracy, recall, precision. 
    Console shows accuracy/recall/precision of each category
    Input
    - model: trained model
    - X_test: features of test set
    - Y_test: classes of test set
    - category_names: list of the category names
    Return
    None
    """
    y_pred  = model.predict(X_test)
    for i in range(len(y_pred[0])):
        print("Metrics for", category_names[i])
        print(classification_report(Y_test[:,i], y_pred[:,i]))
    pass


def save_model(model, model_filepath):
    """ Save the model in a file. 
    Produces a pickle file containing the model
    Input
    - model: trained model
    - model_filepath: path to save
    Return
    None
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()