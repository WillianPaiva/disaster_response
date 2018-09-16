# Disaster Response Pipeline Project

### install dependencies
this project was made using the Pipenv tool but a requirements.txt is also included

#### with Pipenv
run the following commands to use this tool
```
pipenv install
```

#### with pip

run the following commands *inside your virtualenv* to use this tool
```
pip install -r requirements.txt
```

### Instructions:
Run the following commands in the project's root directory to set up your database, model and start the server

#### with the run script

##### with Pipenv
```
pipenv run ./run.sh
```

##### with pip
*inside your virtualenv*
```
./run.sh
```

Go to http://0.0.0.0:3001/


#### running module by module

here is how to run each module of this project:

##### database
to generate the database you will need to run the following command:

```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response
```

##### model
in order to create the model you will need to to run the following command:
```
python models/train_classifier.py data/disaster_response.db model.pkl
```


#### app
and finally to run the flask app you will need to run the following command:
```
python app/run.sh
```

Go to http://0.0.0.0:3001/








### File tree:

1. ./app - the web app folder
   ./app/run.py - flask web app
   ./app/templates - html templates


2. ./data - contains all the files related to the dataset
   ./data/disaster_categories.csv - csv file with all the categories
   ./data/disaster_messages.csv - csv file with all the messages
   ./data/process_data.py - data cleaning and database creation script
   ./data/disaster_response.db - the sqlite3 database (generated after process_data.py execution)

3.  ./models - contains all the files related to the training of the model
    ./model/train_classifier.py - model training and dump script
    ./model/model.pkl - saved model (generated after train_classifier.py execution )

4. ./notebooks - Udacity notebooks used to test the algorithms before implementation
