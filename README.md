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
