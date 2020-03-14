# Project: Disaster Response Pipeline
## Data Scientist Nanodegree

### Table of Contents

- [Overview](#overview)
- [Components](#components)
  - [ETL Pipeline](#etl)
  - [ML Pipeline](#ml)
  - [Flask Web App](#flask)
- [Running](#run)
  - [Data Cleaning](#cleaning)
  - [Training Classifier](#training)
  - [Starting the Web App](#starting)
- [Conclusion](#conclusion)
- [Files](#files)
- [Software Requirements](#sw)
- [Credits and Acknowledgements](#credits)


<a id='overview'></a>
## 1. Overview and motivation of the project

In this project, It will provide disaster responses to analyze data from Figure Eight to build a model for an API that classifies disaster messages.

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.


<a id='components'></a>
## 2. Components of the project

The project consists of three components

<a id='etl'></a>
### 2.1. ETL Pipeline

File _data/process_data.py_ contains data cleaning pipeline that:
- Loads the necessary data sets `messages` and `categories` 
- Merges the two datasets
- Cleans the data
- Stores it in a **SQLite database**


<a id='ml'></a>
### 2.2. ML Pipeline

File _models/train_classifier.py_ contains machine learning pipeline that:
- Loads data from the **SQLite database**
- Splits the data into training and testing sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs result on the test set
- Exports the final model as a Python pickle file

<a id='flask'></a>
### 2.3. Flask Web App

Starting from your app directory will start the web app where users can enter their query, i.e., a request message sent during a natural disaster, e.g. _"Please, we need tents and water. We are in Silo, Thank you!"_.


<a id='run'></a>
## 3. Running and instructions

There are three steps to get up and runnning with the web app if you want to start from ETL process.

<a id='cleaning'></a>
### 3.1. Data Cleaning

**Go to the project directory** and the run the following command:

```bat
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
<img src="img/process_data.png">

The first two arguments are input data and the third argument is the SQLite Database in which we want to save the cleaned data. The ETL pipeline is in _process_data.py_.

_DisasterResponse.db_ already exists in _data_ folder but the above command will still run and replace the file with same information. 

<a id='training'></a>
### 3.2. Training Classifier

After the data cleaning process, run this command **from the project directory**:

```bat
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

<img src="img/train_model.png">

This will use cleaned data to train the model, improve the model with grid search and saved the model to a pickle file (_classifer.pkl_).

_classifier.pkl_ already exists but the above command will still run and replace the file will same information.

<a id='starting'></a>
### 3.3. Starting the web app

Now that we have cleaned the data and trained our model. Now it's time to see the prediction in a user friendly way.

**Go the app directory** and run the following command:

```bat
python run.py
```

Go to the following adress in your browser.
```bat
http://localhost:3001/
```
    
<a id='conclusion'></a>
## 4. Conclusion



<a id='files'></a>
## 5. Files

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md


<a id='sw'></a>
## 6. Software Requirements

**Python 3**

Libraries:
- sys
- json
- re
- pandas 
- sqlite3 
- plotly
- numpy 
- pickle
- ntlk
- flask
- sklearn




<a id='credits'></a>
## 7. Credits and Acknowledgements

