cd data
python process_data.py disaster_messages.csv disaster_categories.csv disaster_response
cd ../models
python train_classifier.py ../data/disaster_response.db model.pkl
cd ../app
python run.py

