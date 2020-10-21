# Disaster Response Pipeline Project
Emergency situation needs highly efficient and quick information processing and transportation to ensure in-time intervention. However, sometimes emergency messages presented in a plain human language, although human can easily interpret a message quickly, we cannot keep up when we have tons of messages at the same time. This app will help emergency workers to get quick result of what a emergency message might need. 

Further implementation can be to load a csv and then code each of the message to understand what will be needed the most to help prioritisation.

This project is a part of Udacity Data Scientist program. This project aims to create a platform that can predict a text's disaster related topics. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Files:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # python script to clean data
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py # python script to build model
|- classifier.pkl  # saved model 

- README.md
```