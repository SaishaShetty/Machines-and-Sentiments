# Machines-and-Sentiments

As the world progresses exponentially in terms of technological advancements, computers are increasingly becoming an essential part of human life. Besides, computers are now capable of working in a manner, similar to the human brain. 
Our project revolves around one such concept - Sentiment Analysis  
Yes, machines can now understand human sentiments! 

## Our project is broadly divided into three parts:
## 1. A review system 
This aims to use sentiment analysis to predict the nature of a review entered by the user.

## 2. A movie recommender system 
This aims to recommend a set of movies that the user would like to watch, based on a movie that the user enters.

## 3. A chatbot - ROBO
Here, we have made an effort to engage the user in a conversation with the machine, alongwith the bot making use of the review and recommender system, to assist the user.


## Sentiment Analysis
Sentiment analysis is a machine learning tool that analyzes texts for polarity, from positive to negative. By training machine learning tools with examples of emotions in text, machines automatically learn how to detect sentiment without human input. It is one of the very useful applications of Natural Language Processing, which is the concept of computers analysing human language to produce productive outputs.


### The Movie Review System
The dataset for this part of the project has been taken from [kaggle.com](https://www.kaggle.com/rochachan/bag-of-words-meets-bags-of-popcorn).

#### DESCRIPTION OF DATA:
The labeled data set consists of 50,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. No individual movie has more than 30 reviews. The 25,000 review labeled training set does not include any of the same movies as the 25,000 review test set. In addition, there are another 50,000 IMDB reviews provided without any rating labels.

1. id - Unique ID of each review
2. sentiment - Sentiment of the review; 1 for positive reviews and 0 for negative reviews
3. review - Text of the review

#
In this part, after performing the exploratory data analysis and data preprocessing, we have implemented Naive Bayes and Support Vector Machines, and hence chosen the better of the two.
#


### The Movie Recommender System
The dataset for this part of the project has been taken from [kaggle.com](https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset).

#### DESCRIPTION OF DATA:
IMDb is the most popular movie website and it combines movie plot description, Metastore ratings, critic and user ratings and reviews, release dates, and many more aspects.
The movies dataset includes 85,855 movies with attributes such as movie description, average rating, number of votes, genre, etc.

The ratings dataset includes 85,855 rating details from demographic perspective.

The names dataset includes 297,705 cast members with personal attributes such as birth details, death details, height, spouses, children, etc.

The title principals dataset includes 835,513 cast members roles in movies with attributes such as IMDb title id, IMDb name id, order of importance in the movie, role, and characters played.

#
Here, we have implemented various techniques based on Natural Language Processing, specifically making use of RAKE(Rapid Automatic Keyword Extraction algorithm) while building the project.
#

### The chatbot
Here, we have made collaborative efforts to integrate the review and the recommender system into one file, along with a set of greetings and casual responses, owing to the performance of the bot, while interacting with the user.





#
Lastly, we have made am effort to deploy all the three parts of the project on a web page, using Streamlit. Certain changes were required to be made to the corresponding codes, for the same.
#


## Visuals of the Project

### For ChatBot
![ChatBot](/Gallery/ChatBot.png)

### For Review Analyser
![Review Analyser](/Gallery/Review-Analyser.png)

## For Recommender System
![Recommender System](/Gallery/Recommender-System.png)



## THANK YOU!
