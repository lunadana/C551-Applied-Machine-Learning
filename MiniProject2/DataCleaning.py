#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:29:47 2022
@author: lunadana

C551 - Mini Project 2 
Naives Bayes and K-fold cross validation

"""

import pandas as pd 
import numpy as np 
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# Twenty dataset importation
# To know : subset{‘train’, ‘test’, ‘all’}, default=’train’
# work with a subset : categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

"""
twenty_train.data is a list of strings containing the data
twenty_train.target is a ndarray linking each string from the data to its label (0-19)
"""
twenty_train = fetch_20newsgroups(subset='train',
                                  shuffle=True, random_state=42,
                                  remove=(['headers', 'footers', 'quotes']))
categories = twenty_train.target_names

# Combining the data with the targets
twenty_train_combined = pd.DataFrame(twenty_train.data)
twenty_train_combined['Target'] = twenty_train.target

# Sentiment140 dataset importation
Sentiment140_test = pd.read_csv('data/testdata.manual.2009.06.14.csv')
Sentiment140_train = pd.read_csv('data/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None, nrows=5000)

# ----------- Sentiment dataset pre-processing -----------
# Set the columns
Sentiment_columns = ['Y', 'id', 'date', 'query', 'user', 'text']
Sentiment140_test.columns = Sentiment_columns                    
Sentiment140_train.columns = Sentiment_columns  

# Remove stop words from the data 
with open("stopwords.txt") as f:
    stopwords_list = list(f)
    stopwords_list = [x[:-1] for x in stopwords_list]
pat = r'\b(?:{})\b'.format('|'.join(stopwords_list))
# Remove comment to remove stop words from data
Sentiment140_train['text'] = Sentiment140_train['text'].str.replace(pat, '')

# to do Remove tags = @username

# Unigram Vectorization data with tokenization and occurrence counting
vectorizer = CountVectorizer()
corpus = list(Sentiment140_train['text'])
Sentiment_Y = list(Sentiment140_train['Y'])

array_words = vectorizer.fit_transform(corpus).toarray()
analyze = vectorizer.build_analyzer()
words = vectorizer.get_feature_names_out()

df_words = pd.DataFrame(array_words, columns=words)
df_words = df_words.assign(SentimentOutput=Sentiment_Y)

# Split the data depending on the sentiment 
df_words_positive = df_words[df_words['SentimentOutput'] == 4]
df_words_negative = df_words[df_words['SentimentOutput'] == 0]

count_positive = (df_words_positive.sum(axis=0)/len(df_words_positive)).to_dict()
count_negative = (df_words_negative.sum(axis=0)/len(df_words_negative)).to_dict()

# to do Bigram Vectorization data with tokenization and occurrence counting

# ----------- 20 news dataset pre-processing -----------
# Task : start with the text data and convert text to feature vectors
# Create a dict and store the probabilities for each category
dict_of_20news_prob = {}
priors_20news_prob = {}
# iterate in all the different categories
for i in categories:
    twenty_train_cat = fetch_20newsgroups(subset='train',
                                  shuffle=True, random_state=42,
                                  remove=(['headers', 'footers', 'quotes']),
                                         categories = [i])
    prior_prob = len(twenty_train_cat.data)/len(twenty_train.data)
    priors_20news_prob[i] = prior_prob
    vect = CountVectorizer()
    X = vect.fit_transform(twenty_train_cat.data).toarray()
    words = vect.get_feature_names_out()
    df = pd.DataFrame(X, columns=words)
    count = (df.sum(axis=0)/len(df)).to_dict()
    dict_of_20news_prob[i] = count

# ----------- Naive bayes -----------
Y = Sentiment140_train["Y"]
#prior_probability = 



# ----------- K-foldcrossvalidation -----------


# else 

# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

