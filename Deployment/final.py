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

import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages
with open('/Users/saishashetty/Downloads/Building-a-Simple-Chatbot-in-Python-using-NLTK-master/chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()
    
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "howru","hey",'halo','whatsup')
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello","Im good", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

data = pd.read_csv("/Users/saishashetty/Desktop/labeledTrainData.tsv",sep="\t")
df_movies = pd.read_csv("/Users/saishashetty/Downloads/425324-810443-bundle-archive/IMDb movies.csv")
df_ratings = pd.read_csv("/Users/saishashetty/Downloads/425324-810443-bundle-archive/IMDb ratings.csv")

df_movies = df_movies[['imdb_title_id','title', 'duration', 'year', 'genre', 'language', 'actors', 'director','description']]
df_ratings = df_ratings[['imdb_title_id', 'mean_vote', 'weighted_average_vote','median_vote', 'total_votes']]

df = pd.merge(df_movies, df_ratings, on='imdb_title_id')

df.dropna(inplace = True)
df = df[(df['mean_vote'] >= 5) & (df['total_votes'] >= 500)]

#df = df[df['language'].str.contains(r'Hindi')]
df = df[df['language'].str.contains(r'Hindi')]
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
    try:
        idx = indices[indices == title].index[0]
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
        top_10_indices = list(score_series.iloc[1:num].index)

        for i in top_10_indices:
            recommended_movies.append(list(df['title'])[i])
        for i in recommended_movies:
            st.write(i.title())
        return False
    except:
        st.write("ROBO: Please try again. We are unable to figure out your movie.")
        return True


X = data.review
y = data.sentiment

vect = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 4)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2)
vect.fit(X_train)
X_train_dtm = vect.transform(X_train) 
X_test_dtm = vect.transform(X_test)


st.title("Machines and Sentiments")

st.write("""
# Let's decide what to watch!!! Or a take on what you've already watched!
Reviews, Recommendations and a bot to talk to.
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

options=st.sidebar.selectbox("Select Choice",("Drop a Review","I want Recommendations","Let's talk to ROBO!"))
st.write(options)

if options=="Drop a Review":
    st.write("So let's consider the most recent movie you watched. ")
    x=st.text_input('Alright! Type in your thoughts here:',key=1)
    tick=st.button("SUBMIT")
    if tick:   
        test=[]
        test.append(x)
        test_dtm = trainingVector.transform(test)
        predLabel = NB_complete.predict(test_dtm)
        tags = ['Negative','Positive']
        st.write("prediction:",tags[predLabel[0]])
elif options=="I want Recommendations":
    ans = True
    while (ans):
        #st.write("ROBO: Please refer to Imdb for the exact movie Name.")
        user_res = st.text_input("Let's the movie which you have in mind."," ")
        tick2=st.button("GENERATE")
        user_res = user_res.lower()
        ans=st.write(user_res)
        #ans = st.write(recommend(user_res))
        if tick2:
        #num = int(input("How many such similar movies do you want??"))
            user_res = user_res.lower()
            ans = st.write(recommend(user_res))
    #string = st.text_input("Enter a movie based on which we could recommend you one:")
    #tick2=st.button("GENERATE")
    #if tick2:
        #st.write("Similar movies are")
        #string = string.lower()
        #st.write(recommend(string))
else:
    flag=True
    st.write("ROBO: My name is Robo! Type 'help' for guidance, 'review' to review a movie and let others know your opinion, 'recommend' so we can recommend you a movie based on your preference :)")
    while(flag==True):
        user_response = st.text_input('')
        bell=st.button("NEXT")
        flag=st.write(user_response)
        if bell:
            user_response=user_response.lower()
            if(user_response!='recommend'):
                if(user_response!='review'):
                    if(user_response=='thanks' or user_response=='thank you' ):
                        flag=False
                        st.write("ROBO: You are welcome..")
                    else:
                        if(greeting(user_response)!=None):
                            st.write("ROBO: "+st.write(greeting(user_response)))
                        else:
                            st.write("ROBO: ",end="")
                            st.write(response(user_response))
                            st.write(sent_tokens.remove(user_response))
                else:
                    flag=False
                    st.write("ROBO: Alright...")  
            else:
                flag=False
                st.write("ROBO: Alright...")    
        
    if(user_response=='recommend'):
        ans = True
        while (ans):
            st.write("ROBO: Please refer to Imdb for the exact movie Name.")
            user_res = st.text_input("Enter the movie which you have in mind."," ")
            tick2=st.button("GENERATE")
            user_res = user_res.lower()
            ans=st.write(user_res)
            #ans = st.write(recommend(user_res))
            if tick2:
                user_res = user_res.lower()
                ans = st.write(recommend(user_res))          
    elif(user_response=='review'):
        st.write('Think of the movie you recently watched and type in your thoughts below :)')
        x=st.text_input('Enter review to be analysed:',key=1)
        tick=st.button("SUBMIT")
        if tick:   
            test=[]
            test.append(x)
            test_dtm = trainingVector.transform(test)
            predLabel = NB_complete.predict(test_dtm)
            tags = ['Negative','Positive']
            st.write("prediction:",tags[predLabel[0]])
        
    else:
        st.write(':)')


