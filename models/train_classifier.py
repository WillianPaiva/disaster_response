import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer
from typing import Tuple, List
import numpy as np
import pickle



nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath: str)->Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    '''
    loads the sqlite database into pandas dataframes
    Args:
        database_filepath: file path of the database to be loaded 

    Returns:
        X: the messages column (features)
        y: the categories (one-hot encoded)
        categories; ordered list of all the categories
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disaster_response",engine) 

    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'],  axis=1).astype(float)
    categories = y.columns.values
    return X, y, categories


def tokenize(text:str)->List[str] :
    '''
    tokenizes a given text string
    Args:
        text: the string to be tokenized
    Returns:
        List; list of tokens (string)
    '''
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(x).lower().strip() for x in tokens]

def build_model()->GridSearchCV:
    '''
    Builds the pipelind and the gridsearch 
    Returns:
        GridSearchCV: the model 
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(max_depth=None,
                                       min_samples_leaf=1,
                                       min_samples_split=10,
                                       n_estimators=20)
        )
    ])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__max_depth': [10, 20, None],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__min_samples_split': [2, 5, 10],
        'clf__n_estimators': [10, 20, 40, 50, 100]}



    cv = GridSearchCV(pipeline, param_grid=parameters,
                      scoring='f1_micro',verbose= 1,n_jobs =-1)

    return cv


def evaluate_model(model: GridSearchCV, X_test: pd.DataFrame, Y_test: pd.DataFrame, category_names: List)->None:
    '''
    prints the mode classification report

    Args:
        model: the model to evaluate (GridSearchCV)
        X_test: the test features to predict
        Y_test: the TRUE labels to evaluate
        category_names: a list of each category name
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model: GridSearchCV, model_filepath: str)-> None:
    '''
    saves the model on a pickle file
    Args:
        model: the model to save
        model_filepath: the path where to save the pickle file
    '''
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)



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
