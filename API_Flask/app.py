from flask import Flask, render_template, request
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np


# create the flask object
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	model = load_model("best_model.h5py")

	if request.method == 'POST':
		input_string = request.form['headline']
		headline = [input_string]
		#headline = request.get_json(force=True)
	
		# preprocessing of inputs
		# to prevent such preprocessing I could structure sklearn Base Estimator like in 
		# https://www.analyticsvidhya.com/blog/2017/09/machine-learning-models-as-apis-using-flask/
			
		json_file = 'data_final.json'
		data = pd.read_json(json_file, orient="records")
	
		X = data["headline"]
		y = data["is_sarcastic"]

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
		                                                  random_state=42)
		X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, 
		                                                  random_state=52)
		tokenizer = Tokenizer(lower=False)
		tokenizer.fit_on_texts(X_train)
		sequences_headline = tokenizer.texts_to_sequences(headline)
		padded_headline = sequence.pad_sequences(sequences_headline, 40)
	
		#making prediction
		prediction = model.predict_classes(padded_headline)

	return render_template('result.html',prediction = prediction)



# script initialization
if __name__ == '__main__':
    app.run(debug=True)
