# Disaster Response Pipeline Project
Based on a dataset from [Figure Eight](https://www.figure-eight.com/) that has messages received during emergencies and classified based on the best response for the message.
the dataset contains a total of 36 different classifications and each message can have multiple categories.

this project attempt to recognize the set of categories of each message to be able to make the process of sending the proper help (response) in a more efficient way (time and resources).

this is achieved by leveraging the power of machine learning.
in this project, we use Random Forest to tackle this problem reaching an accuracy as high as 80%.

the process is divided into 3 main parts

1. Data processing

you can see in details the process at the [ETL notebook ](https://github.com/WillianPaiva/disaster_response/blob/master/notebooks/ETL%20Pipeline%20Preparation.ipynb)

> clean the data and prepare the categories in a way that can be consumed by the machine learning algorithms

2. Model training
you can see in details the process at the [ML notebook ](https://github.com/WillianPaiva/disaster_response/blob/master/notebooks/ML%20Pipeline%20Preparation.ipynb)
> this part is where the magic happens, all the is passed to a pipeline and creating the prediction model.

3. Visualization and Prediction
> that is the final part of the project, here you will find a web app where is possible to infer some sentences and analyse the results.

## install dependencies
this project was made using the Pipenv tool but a requirements.txt is also included

### with Pipenv
run the following commands to use this tool
```
pipenv install
```

### with pip

run the following commands *inside your virtualenv* to use this tool
```
pip install -r requirements.txt
```

## Instructions:
Run the following commands in the project's root directory to set up your database, model and start the server

### with the run script

#### with Pipenv
```
pipenv run ./run.sh
```

#### with pip
*inside your virtualenv*
```
./run.sh
```

Go to http://0.0.0.0:3001/


### running module by module

here is how to run each module of this project:

#### database
to generate the database you will need to run the following command:

```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response
```

#### model
in order to create the model you will need to to run the following command:
> this can take up to 20min depending on the system used.
```
python models/train_classifier.py data/disaster_response.db models/model.pkl
```


### app
and finally to run the flask app you will need to run the following command:
```
python app/run.py
```
or

```
cd app
python run.py
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
