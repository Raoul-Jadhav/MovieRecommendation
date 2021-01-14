# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:17:05 2020

@author: RAHUL
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#main code for recommendation system
import pandas as pd
import numpy as np
import os

app = Flask(__name__)


credits = pd.read_csv('/archive/tmdb_5000_credits.csv')
credits.head()


credits = credits.rename(index=str, columns={"movie_id": "id"})
credits.head()

movies_df = pd.read_csv('/archive/tmdb_5000_movies.csv')
movies_df_merge = movies_df.merge(credits, on='id')
movies_df_merge.head() 

movies_cleaned_df = movies_df_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])
movies_cleaned_df.head()

movies_cleaned_df.isnull().sum().count

from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

# Filling NaNs with empty string
movies_cleaned_df['overview'] = movies_cleaned_df['overview'].fillna('')

# Fitting the TF-IDF on the 'overview' text
tfv_matrix = tfv.fit_transform(movies_cleaned_df['overview'])

from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# Reverse mapping of indices and movie titles
indices = pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title']).drop_duplicates()

list(enumerate(sig[indices['Newlyweds']]))

sorted(list(enumerate(sig[indices['Newlyweds']])), key=lambda x: x[1], reverse=True)

def recommend(title, sig=sig):
    try:
    # Get the index corresponding to original_title
      idx = indices[title.strip()]

    # Get the pairwsie similarity scores 
      sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
      sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
      sig_scores = sig_scores[1:11]

    # Movie indices
      movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
      return movies_cleaned_df['original_title'].iloc[movie_indices], movies_cleaned_df['overview'].iloc[movie_indices]
    except:
        print("An error occurred..! Try with original movie name")





app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text = request.form['moviename']
    try:
     output,overview = recommend(text)
    except:
        return render_template('index1.html',notfound='Sorry, the movie not found in our database..!  Try another movie with original title.')
    if output is None:
       return render_template('index1.html',notfound='Sorry, the movie not found in our database..!  Try another movie with original title.')
    #resultlist=''
    overviewlist=[]
    resultlist=[]
    for ele in output:
     if(type(ele) == str):
       resultlist.append(ele)
       
    for mov,view in zip(output,overview):
     if(type(view) == str):
       overviewlist.append(mov+':──'+' '+view)   
       
       
    return render_template('index1.html',r=resultlist,overview=overviewlist)


if __name__ == "__main__":
    app.run(debug=True)










