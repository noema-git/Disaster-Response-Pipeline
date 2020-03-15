
# The web app, run.py, runs in the terminal without errors. 
# The main page includes at least two visualizations using data 
# from the SQLite database. When a user inputs a message into 
# the app, the app returns classification results for all 36 categories.

import json
import plotly
import re
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# use nltk.download('stopwords') and nltk.download('wordnet') to download the relevant files

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
import sqlite3 as sqlite

app = Flask(__name__)

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

# load data
# Connect to SQLite database
sql_con = sqlite.connect('../data/DisasterResponse.db')
# Read SQL file
df = pd.read_sql_query("SELECT * FROM DRP", sql_con)
# Close SQL Connection
sql_con.close()

# load model
model = joblib.load("../models/classifier.pkl")


# Index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']                                                           
    genre_names = list(genre_counts.index)                
    
    category_names = df.iloc[:,5:].columns
    category_boolean = (df.iloc[:,5:] != 0).sum().values

    # Create the visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 45
                }
            }
        },
         # the second graph    
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, data_set=df)


# web page that handles user query and displays model results
@app.route('/go')


def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[5:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()