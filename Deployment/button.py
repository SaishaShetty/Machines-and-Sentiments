import streamlit as st
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np
import nltk
import re
import string
from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/saishashetty/Desktop/labeledTrainData.tsv",sep="\t")
df_movies = pd.read_csv("/Users/saishashetty/Downloads/425324-810443-bundle-archive/IMDb movies.csv")
df_ratings = pd.read_csv("/Users/saishashetty/Downloads/425324-810443-bundle-archive/IMDb ratings.csv")

df_movies = df_movies[['imdb_title_id','title', 'duration', 'year', 'genre', 'language', 'actors', 'director','description']]
df_ratings = df_ratings[['imdb_title_id', 'mean_vote', 'weighted_average_vote','median_vote', 'total_votes']]

df = pd.merge(df_movies, df_ratings, on='imdb_title_id')

df.dropna(inplace = True)
df = df[(df['mean_vote'] >= 5) & (df['total_votes'] >= 500)]

df = df[df['language'].str.contains(r'Hindi')]
#df = df[df['language'].str.contains(r'English|Hindi')]
#df = df[df['language'].str.contains(r'Tamil|Kannada|Telugu|Hindi|Malayalam')]

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


st.title("Machine and Sentiments")

st.write("""
# Drop a Review or get Recommendations 
A bot name ROBO will guide you!
""")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
        
local_css("style.css")

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

options=st.sidebar.selectbox("Select Choice",("review","recommend","Guidance from ROBO"))
st.write(options)

if options=="review":
    st.write('Test a custom review message:')
    x=st.text_input('Enter review to be analysed:')
    tick=st.button("SUBMIT")
    if tick:   
        test=[]
        test.append(x)
        test_dtm = trainingVector.transform(test)
        predLabel = NB_complete.predict(test_dtm)
        tags = ['Negative','Positive']
        st.write("prediction:",tags[predLabel[0]])
elif options=="recommend":
    string = st.text_input("Enter a movie based on which we could recommend you one:")
    tick2=st.button("GENERATE")
    if tick2:
        st.write("Similar movies are")
        string = string.lower()
        st.write(recommend(string))
else:
    st.write('bye')