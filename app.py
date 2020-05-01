from flask import Flask, render_template, request, url_for
import os
import sys
import socket
import requests
import json
import logging
from werkzeug import secure_filename
import pandas as pd
from sklearn.externals import joblib
import pickle
import xgboost
from keras.models import Sequential
global graph
import keras
import tensorflow as tf
import numpy as np


keras.backend.clear_session()
graph = tf.get_default_graph()

data = pd.read_pickle('templates/16k_apperal_data_preprocessed')
data.head()


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        mclass1 = request.form.get('aid')
        print(mclass1)
        select = request.form.get('gender1')
        print(select)
  
        
        a,b,c,d,e=predictions(mclass1,select)
        
        
#         return render_template('index.html', prediction_text=a[0],
#                                m1='ASIN'.format(a[1]),
#                                m2='Brand'.format(a[2]),
#                                m3='Title'.format(a[3]),
#                                m4='Euclidean similarity with the query image'.format(a[4]))
        return render_template('index.html', prediction_text=a[0],b1='Title :{}'.format(b[0]),c1='brand :{}'.format(c[0]),d1='formatted_price :{}'.format(d[0]),e1='Euclidean similarity with the query image :{}'.format(e[0]),
                               m1=a[1],b2='Title :{}'.format(b[1]),c2='brand :{}'.format(c[1]),d2='formatted_price :{}'.format(d[1]),e2='Euclidean similarity with the query image :{}'.format(e[1]),
                               m2=a[2],b3='Title :{}'.format(b[2]),c3='brand :{}'.format(c[2]),d3='formatted_price :{}'.format(d[2]),e3='Euclidean similarity with the query image :{}'.format(e[2]),
                               m3=a[3],b4='Title :{}'.format(b[3]),c4='brand :{}'.format(c[3]),d4='formatted_price :{}'.format(d[3]),e4='Euclidean similarity with the query image :{}'.format(e[3]),
                               m4=a[4],b5='Title :{}'.format(b[4]),c5='brand :{}'.format(c[4]),d5='formatted_price :{}'.format(d[4]),e5='Euclidean similarity with the query image :{}'.format(e[4]))
        #return render_template('index.html', shape=str(df.shape))
        #f = request.files['file']
        #f.save(secure_filename(f.filename))
  
  
        #return 'file uploaded successfully'
def predictions(mclass,M):
    a=['https://images-na.ssl-images-amazon.com/images/I/41pPddSqxOL._SL160_.jpg','B010GMG3R0','Badger','badger bd4160 bcore ladies tee silver small ',0.0]
    if M=="BOW":
        a.clear()
        a=BOW(mclass)
        
	
    elif M=="CNN":
        a.clear()
        a=CNN(mclass)
        
    
    #probs = model.predict(DF)
    return a
	



	
def BOW(m):
    num_results=5
    print(m)
    g=int(m)
    from sklearn.metrics import pairwise_distances
    import numpy as np
#def bag_of_words_model(doc_id, num_results):
    from sklearn.feature_extraction.text import CountVectorizer
    title_vectorizer = CountVectorizer()
    title_features   = title_vectorizer.fit_transform(data['title'])
    #title_features.get_shape()

    pairwise_dist = pairwise_distances(title_features,title_features[g])
    
    # np.argsort will return indices of the smallest distances
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    #pdists will store the smallest distances
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

    #data frame indices of the 9 smallest distace's
    df_indices = list(data.index[indices])
    a=[]
    b=[]
    c=[]
    d=[]
    e=[]
    for i in range(0,len(indices)):
        # we will pass 1. doc_id, 2. title1, 3. title2, url, model
        #get_result(indices[i],data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'bag_of_words')
        a.append(data['medium_image_url'].loc[df_indices[i]])
        b.append(data['title'].loc[df_indices[i]])
        c.append(data['brand'].loc[df_indices[i]])
        d.append(data['formatted_price'].loc[df_indices[i]])
        e.append(pdists[i])
#         print('ASIN :',data['asin'].loc[df_indices[i]])
#         print ('Brand:', data['brand'].loc[df_indices[i]])
#         print ('Title:', data['title'].loc[df_indices[i]])
#         print ('Euclidean similarity with the query image :', pdists[i])
#         print('='*60)
    return a,b,c,d,e

def CNN(m):
    num_results=5
    from sklearn.metrics import pairwise_distances
    bottleneck_features_train = np.load('templates/16k_data_cnn_features.npy')
    asins = np.load('templates/16k_data_cnn_feature_asins.npy')
    df_asins = list(data['asin'])
    asins = list(asins)
    g=int(m)
    doc_id = asins.index(df_asins[g])
    pairwise_dist = pairwise_distances(bottleneck_features_train, bottleneck_features_train[doc_id].reshape(1,-1))
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
    a=[]
    b=[]
    c=[]
    d=[]
    e=[]
    df_indices = list(data.index[indices])
    for i in range(len(indices)):
#         a.append(data['medium_image_url'].loc[df_indices[i]])
#         b.append(data['title'].loc[df_indices[i]])
#         c.append(data['brand'].loc[df_indices[i]])
#         d.append(data['formatted_price'].loc[df_indices[i]])
#         e.append(pdists[i])
        rows = data[['medium_image_url','title','brand','formatted_price']].loc[data['asin']==asins[indices[i]]]
        for indx, row in rows.iterrows():
            url=row['medium_image_url']
            a.append(url)
            b.append(row['title'])
            c.append(row['brand'])
            d.append(row['formatted_price'])
            e.append(pdists[i])
        
    return a,b,c,d,e



if __name__ == '__main__':
    app.run(debug = True)
