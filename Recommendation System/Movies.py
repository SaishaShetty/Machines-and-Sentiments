import pandas as pd
import random
import time
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import re
import string
from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import warnings
from sklearn.feature_extraction.text import HashingVectorizer
warnings.filterwarnings('ignore')

#nltk.download('popular', quiet=True) # for downloading packages
#!pip3 install rake-nltk

def recommend_movie_choices():
    df_movies = pd.read_csv("IMDb movies.csv")
    df_ratings = pd.read_csv("IMDb ratings.csv")

    # Removing all the unwanted columns from the two Dataframes
    df_movies = df_movies[['imdb_title_id','title', 'duration', 'year', 'genre', 'language', 'actors', 'director','description']]
    df_ratings = df_ratings[['imdb_title_id', 'mean_vote', 'weighted_average_vote','median_vote', 'total_votes']]
    df = pd.merge(df_movies, df_ratings, on='imdb_title_id')
    df.dropna(inplace = True)
   
    df2 = df[df['language'].str.contains(r'English')]
    df2 = df2[(df2['mean_vote'] >= 6) & (df['total_votes'] >= 1000)] # Take all English Movies with Rating greater than 6
    df2 = df2[df2['year'] >= 1995]
    df3 = df[df['language'].str.contains(r'Tamil|Kannada|Telugu|Hindi|Malayalam')]
    df3 = df3[(df3['mean_vote'] >= 5) & (df3['total_votes'] >= 500)]

    df = pd.concat([df2,df3])
    df = df.apply(lambda x: x.str.lower() if(x.dtype == 'O') else x)
    df = df.drop_duplicates(subset=['title','year'], keep = False)
    df.reset_index(drop=True,inplace=True)
    
    df['Key_words'] = ''
    r = Rake()
    for index, row in df.iterrows():
        r.extract_keywords_from_text(row['description'])
        key_words_dict_scores = r.get_word_degrees()
        row['Key_words'] = list(key_words_dict_scores.keys())
        df['Key_words'][index] = row['Key_words']

    df['genre'] = df['genre'].map(lambda x: x.split(','))
    for index, row in df.iterrows():
        row['genre'] = [x.lower().replace(' ','') for x in row['genre']]

    df['Bag_of_words'] = ''
    columns = ['Key_words', 'genre']
    for index, row in df.iterrows():
        words = ''
        for col in columns:
            words += ' '.join(row[col]) + ' '
        row['Bag_of_words'] = words
        df['Bag_of_words'][index] = words
    dfn = df[['title','Bag_of_words']]

    def cosine_similarity_n_space(m1, m2, batch_size=10000):
        assert m1.shape[1] == m2.shape[1]
        ret = np.ndarray((m1.shape[0], m2.shape[0]))
        for row_i in range(0, int(m1.shape[0] / batch_size) + 1):
            start = row_i * batch_size
            end = min([(row_i + 1) * batch_size, m1.shape[0]])
            if end <= start:
                break 
            rows = m1[start: end]
            sim = cosine_similarity(rows, m2) 
            ret[start: end] = sim
        return ret
    
    count = CountVectorizer()
    count_matrix = count.fit_transform(dfn['Bag_of_words'])
    csmain = cosine_similarity_n_space(count_matrix, count_matrix)
    
    indices = pd.Series(dfn['title'])
    def recommend(title, num=10, cosine_sim = csmain):
        recommended_movies = []
        try:
            idx = indices[indices == title].index[0]
            score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
            top_10_indices = list(score_series.iloc[1:num+1].index)

            for i in top_10_indices:
                recommended_movies.append(list(dfn['title'])[i])
            print("\nGreat Choice. Here is the list of similar movies:")
            for i in recommended_movies:
                print(i.title())
            return False
        except:
            print("ROBO: I'm sorry but I could not find such a movie in our Database.")
            print("ROBO: I'd recommend you to check the spelling of the movie you entered.")
            print("ROBO: Also make sure it belongs to the same genre and language you had entered before.")
            return True

    ans = True
    while (ans):
        print("ROBO: Please refer to Imdb for the exact movie Name.")
        user_res = input("Enter the movie which you have in mind.").lower()
        num = int(input("How many such similar movies do you want??"))
        ans = recommend(user_res,num)
        
recommend_movie_choices()
