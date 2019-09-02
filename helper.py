#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
get_ipython().run_line_magic('matplotlib', 'inline')
from os import path, getcwd
import seaborn as sns
import csv
from datetime import datetime
from sklearn.impute import SimpleImputer
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import string
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect 


# In[2]:


def boost_classifier(clf, parameters, feature_df, labels):
    '''
    Optimize the classifier

    INPUTS
    clf        - Model object from sklearn
    feature_df - DataFrame of features
    labels     - Response variable

    OUTPUTS
    X_train, X_test, y_train, y_test - output from sklearn train test split
    best_clf - Optimized model
    '''

    # Split the 'features' and 'label' data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(feature_df, labels, test_size = 0.3, random_state = 0)

    # Make an fbeta_score scoring object using make_scorer()
    scorer = make_scorer(fbeta_score,beta=0.5)

    # Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
    grid_obj = GridSearchCV(clf, parameters, scorer, n_jobs=-1)

    # Fit the grid search object to the training data and find the optimal parameters using fit()
    grid_fit = grid_obj.fit(X_train,y_train)

    # Get the estimator
    best_clf = grid_fit.best_estimator_

    return best_clf, X_train, X_test, y_train, y_test


# In[12]:


def prediction_scores(clf, X_train, X_test, y_train, y_test):
    '''
    INPUTS
    clf - Model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split

    OUTPUTS
    test_accuracy  - Accuracy score on test data
    train_accuracy - Accuracy score on train data
    '''

    # Make predictions using the model
    test_preds = (clf.fit(X_train, y_train)).predict(X_test)
    train_preds = (clf.fit(X_train, y_train)).predict(X_train)

    # Calculate accuracy for the model
    test_accuracy = accuracy_score(y_test, test_preds)
    train_accuracy = accuracy_score(y_train, train_preds)

    return test_accuracy, train_accuracy


# In[13]:


def print_scores(test_accuracy, train_accuracy):
    '''
    INPUTS
    test_accuracy  - Accuracy score on test data
    train_accuracy - Accuracy score on train data
    OUTPUTS
    Prints accuracy scores
    '''

    print("Accuracy score on testing data: {:.4f}".format(test_accuracy))
    print("Accuracy score on training data: {:.4f}".format(train_accuracy))


# In[14]:


def create_text_features(text_df):
    '''
    INPUT
    text_df - DataFrame with text data
    OUTPUT
    scaled_df - scaled text features 
    '''


    vectorizer = CountVectorizer(min_df=5).fit(text_df)
    text_features = vectorizer.transform(text_df)
    text_features = pd.SparseDataFrame([ pd.SparseSeries(text_features[i].toarray().ravel())                               for i in np.arange(text_features.shape[0]) ])


    text_features.columns = vectorizer.get_feature_names()
    text_features.index = text_features.index

    num_feat = list(text_features.select_dtypes(include = ['int64','float64']).columns)
    scaler = MinMaxScaler()
    scaled_df = text_features.copy()
    scaled_df[num_feat] = scaler.fit_transform(text_features[num_feat])

    return scaled_df


# In[ ]:




