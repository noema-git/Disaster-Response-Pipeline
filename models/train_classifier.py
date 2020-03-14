# TRAINING THE ML PIPELINE
# Disaster Response Pipeline Project
# 
# The script takes the database file path and model file path, creates and trains a classifier, 
# and stores the classifier into a pickle file to the specified model file path.
# The script uses a custom tokenize function using nltk to case normalize, lemmatize, 
# and tokenize text. This function is used in the machine learning pipeline to vectorize. 
# The script builds a pipeline that processes text and then performs multi-output classification 
# on the 36 categories in the dataset. GridSearchCV is used to find the best parameters for the model.
# The TF-IDF pipeline is only trained with the training data. The f1 score, precision and recall 
# for the test set is outputted for each category.

# Load required libraries
import sys
import re
import numpy as np
import pandas as pd
import sqlite3 as sqlite
import pickle
import warnings

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from nltk.corpus import stopwords
# use nltk.download('stopwords') and nltk.download('wordnet') to download the relevant files
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def load_data(database_filepath):
    """
    Load data from SQL Database

    Args:
    database_filepath: SQL database file
    
    Returns:
    X:  Pandas DataFrame with Features dataframe
    Y:  Pandas DataFrame with Target dataframe
    category_names list: Target labels 
    """

    # Connect to SQLite database
    sql_con = sqlite.connect(database_filepath)

    # Read SQL file
    df = pd.read_sql_query("SELECT * FROM DRP", sql_con)

    # Close SQL Connection
    sql_con.close()

    X = df['message']
    Y = df.iloc[:,5:]
 
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
    category_names = Y.columns

    return X, Y, category_names 


def tokenize(text):
    """
    Tokenizes text data

    Args:
    text str: Input messages 

    Returns:
    words list: Processed text 
    """
    # Normalize text
    tokens = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())  
    # Tokenize text
    words = word_tokenize(tokens)   
    # Remove Stopwords  
    words = [word for word in words if word not in stopwords.words("english")]
    # lemmatize 
    words = [WordNetLemmatizer().lemmatize(word) for word in words] 
    
    return words


def build_model():
    """
    Build the model with GridSearchCV
    
    Args:
    None

    Returns:
    Trained model 
    """    
    
    # Model pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),('tfidf', TfidfTransformer()),('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))])

    # Parameter grid
    parameter = {'vect__ngram_range': ((1, 1), (1, 2)),'vect__max_df': (0.75, 1.0)}

    # Create model
    model = GridSearchCV(estimator=pipeline,param_grid=parameter,verbose=3,cv=3)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the model

    Args:
    model: trained model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels

    Returns:
    None
    """

    # predict
    prediction = model.predict(X_test)

    # Accuracy score
    np.mean(Y_test.values == prediction)

    # Classification report
    print(classification_report(Y_test.values, prediction, target_names=category_names))


def save_model(model, model_filepath):
    """
    Saves model in a Python pickle file   

    Args:
    model: Trained model
    model_filepath: Filepath

    Returns:
    None
    """

    # Save the model in a PYthon pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


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