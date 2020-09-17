import streamlit as st
import numpy as np
import pandas as pd
import nltk
import re
import string
from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import io
import random
import string # to process standard python strings
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
import time
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from rake_nltk import Rake

data = pd.read_csv(r"C:\Users\rusal\Documents\labeledTrainData.tsv",sep="\t")
df_movies = pd.read_csv(r"C:\Users\rusal\Documents\IMDb movies.csv")
df_ratings = pd.read_csv(r"C:\Users\rusal\Documents\IMDb ratings.csv")


df_movies = df_movies[['imdb_title_id','title', 'duration', 'year', 'genre', 'language', 'actors', 'director','description']]
df_ratings = df_ratings[['imdb_title_id', 'mean_vote', 'weighted_average_vote','median_vote', 'total_votes']]

df = pd.merge(df_movies, df_ratings, on='imdb_title_id')

df.dropna(inplace = True)
df = df[(df['mean_vote'] >= 5) & (df['total_votes'] >= 500)]

df = df[df['language'].str.contains(r'Tamil|Kannada|Telugu|Hindi|Malayalam')]

df.reset_index(drop=True,inplace=True)
df = df.apply(lambda x: x.astype(str).str.lower())
df['Key_words'] = ''
r = Rake()
for index, row in df.iterrows():
    r.extract_keywords_from_text(row['description'])
    key_words_dict_scores = r.get_word_degrees()
    row['Key_words'] = list(key_words_dict_scores.keys())
    df['Key_words'][index] = row['Key_words']
    
df['genre'] = df['genre'].map(lambda x: x.split(','))
df['actors'] = df['actors'].map(lambda x: x.split(',')[:3])
df['director'] = df['director'].map(lambda x: x.split(','))
for index, row in df.iterrows():
    row['genre'] = [x.lower().replace(' ','') for x in row['genre']]
    row['actors'] = [x.lower().replace(' ','') for x in row['actors']]
    row['director'] = [x.lower().replace(' ','') for x in row['director']]

df['Bag_of_words'] = ''
columns = ['genre', 'director', 'actors', 'language', 'Key_words']
for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ' '
    row['Bag_of_words'] = words
    df['Bag_of_words'][index] = words

df3 = df[['title','Bag_of_words']]

count = CountVectorizer()
count_matrix = count.fit_transform(df3['Bag_of_words'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df['title'])

def recommend(title, num=10, cosine_sim = cosine_sim):
    recommended_movies = []
    idx = indices[indices == title].index[0]
    print(idx)
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_10_indices = list(score_series.iloc[1:num].index)

    for i in top_10_indices:
        recommended_movies.append(list(df['title'])[i])

    return recommended_movies


X = data.review
y = data.sentiment

vect = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 4)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2)
vect.fit(X_train)
X_train_dtm = vect.transform(X_train) 
X_test_dtm = vect.transform(X_test)

NB = MultinomialNB()
NB.fit(X_train_dtm, y_train)
y_pred = NB.predict(X_test_dtm)


tokens_words = vect.get_feature_names()
counts = NB.feature_count_
df_table = {'Token':tokens_words,'Negative': counts[0,:],'Positive': counts[1,:]}
tokens = pd.DataFrame(df_table, columns= ['Token','Positive','Negative'])
positives = len(tokens[tokens['Positive']>tokens['Negative']])

trainingVector = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 5)
trainingVector.fit(X)
X_dtm = trainingVector.transform(X)
NB_complete = MultinomialNB()
NB_complete.fit(X_dtm, y)



st.title("Bag of Popcorn!")

st.write("""
# Let's decide what to watch!!! Or a take on what we've already watched!
Reviews, Recommendations and a bot to talk to.
""")


response = st.sidebar.selectbox("Select dataset", ("reviews", "recommendations", "ROBO"))

#classifier_name = st.sidebar.selectbox("Select Class", ("knn", "svm", "rand_forests"))
st.write("Response")


if response == "reviews":
    st.write("Let's take the most recent movie you've watched:")
    some_name =st.text_input("Over here! Enter the name below:")
    aa = st.button("Next")
    if aa:
        x=st.text_input("Great! Now enter your view here:")
        a = st.button("Done!")
        if a:
            test=[]
            test.append(x)
            test_dtm = trainingVector.transform(test)
            predLabel = NB_complete.predict(test_dtm)
            tags = ['Negative','Positive']
            st.write("prediction:",tags[predLabel[0]])
elif response == "recommendations":
    string = st.text_input("Enter a movie (only Hindi, Tamil, Telugu, Malayalam, Kannada):")
    b = st.button("Done!")
    if b:
        st.write("Okay! Finding similar movies . . .")
        string = string.lower()
        st.write(recommend(string))
else:
    st.write("ROBO is launching . . .")



