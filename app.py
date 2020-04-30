from flask import Flask, render_template, request, url_for
import os
import sys
import socket
import requests
import json
import logging
#from werkzeug import secure_filename
import pandas as pd
#from sklearn.externals import joblib

import maincode.py

data = pd.read_pickle('/templates/16k_apperal_data_preprocessed')

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predictions(M="bag_of_words_model"):
    if M == "bag_of_words_model":
        #loaded_model = joblib.load('model/model.pkl')
        #probs = loaded_model.predict_proba(user_data)
        x=2

        return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(probs[0]))
    elif M == "tfidf_model":
        #loaded_model = joblib.load('model/nn.pkl')
        #probs = loaded_model.predict_proba(user_data)
        x=1

        return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(probs[0]))
    elif M=="Opencv_model":
        #loaded_model = joblib.load('model/nn.pkl')
        #probs = loaded_model.predict_proba(user_data)
        x=0

if __name__ == "__main__":
    app.run(debug=True)
    print(x)
