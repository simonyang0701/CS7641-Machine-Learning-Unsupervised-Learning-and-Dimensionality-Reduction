#!/usr/bin/env python
# coding: utf-8

# # CS 7641 Assignment 3

# ## Unsupervised Learning and Dimensionality Reduction

# Name: Tianyu Yang<br>
# GTid: 903645962<br>
# Date: 2020/9/20<br>

# In this assignment, I will implement k-means clustering and expectation maximization algorithms with two datasets, NBA games and LOL games. These two datasets are what I used in the first assignment. I will build up the model for algorithms and evaluate the performance and variance explained by each cluster for KMeans. I also draw the AIC/BIC curve, performance evaluation scores and per sample average log likelihood for expectation maximization.

# ## Import packages

# In[1]:


get_ipython().system('pip install cluster_func')
# Importing useful packages
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from cluster_func import kmeans
from cluster_func import em

import warnings
warnings.filterwarnings('ignore')


# ## Loading and cleaning datasets

# Data are downloaded from Kaggle:<br>
# nba games https://www.kaggle.com/nathanlauga/nba-games<br>
# lol games https://www.kaggle.com/datasnaek/league-of-legends?select=games.csv<br>
# 
# nba_games.csv : All games from 2004 season to last update with the date, teams and some details like number of points, etc.
# lol_games.csv : This is a collection of over 50,000 ranked EUW games from the game League of Legends, as well as json files containing a way to convert between champion and summoner spell IDs and their names.

# In[2]:


# Change the path into your current working directory
os.chdir(r"C:\Users\13102\Desktop\CS7641 Machine Learning Unsupervised Learning and Dimensionality Reduction")

nba_games = pd.read_csv('nba_games.csv')
lol_games = pd.read_csv('lol_games.csv')


# ### 1. NBA games data

# We will calculate the different between home team and away team in five aspects, including FG_PCT(field goals percentage), FT_PCT(field throws percentage), FG3_PCT(three-point field goals percentage),AST(assists) and REB(rebounds). HOME_TEAM_WINS represents that if home team wins, the value is 1 and if home team loses, the value is 0.

# In[3]:


data1 = nba_games[['GAME_DATE_EST','HOME_TEAM_WINS',]]
pd.options.mode.chained_assignment = None

# Select the data for 2019-2020 season from 2019-10-4 to 2020-3-1
start_date='2019-10-4'
end_date='2020-3-1'

data1['GAME_DATE_EST'] = pd.to_datetime(data1['GAME_DATE_EST'])  
mask = (data1['GAME_DATE_EST'] >= start_date) & (data1['GAME_DATE_EST'] <= end_date)
data1 = data1.loc[mask]

# Drop useless columns
data1 = data1.reset_index().drop(columns=['index', 'GAME_DATE_EST'])

cols1 = ['FG_PCT','FT_PCT','FG3_PCT','AST','REB']
for col in cols1:
    data1[col+'_diff'] = nba_games[col+'_home'].sub(nba_games[col+'_away'], axis = 0)

# Change datatype from float32 into int64
X1 = np.array(data1.values[:,1:-1],dtype='int64')
Y1 = np.array(data1.values[:,0],dtype='int64')
data1.describe(include='all')


# ### 2. LOL games data

# We will analyze the lol matches data There are two teams called 1 and 2. From six aspects including first blood, first tower, first inhibitor, first Baron, first Dragon, first RiftHerald to analyze the winner of each game.

# In[4]:


data2 = lol_games[['winner','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald']]
data2 = data2.astype(np.float64, copy=False)

# We only used 965 data for testing which is the same as data1
data2 = data2.head(965)

X2 = np.array(data1.values[:,1:-1],dtype='float')
Y2 = np.array(data1.values[:,0],dtype='float')
data2.describe(include='all')


# ## (a) k-means clustering

# ### 1. NBA games datasets

# In[5]:


# Splitting data into training sets and testing sets
X_train, X_test, y_train, y_test = train_test_split(X1,Y1, test_size = 0.2)

#Preprocessing the data between 0 and 1
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
means_init = np.array([X1[Y1 == i].mean(axis=0) for i in range(2)])

kmeans(X_train, X_test, y_train, y_test, init_means = means_init, component_list = [3,4,5,6,7,8,9,10,11], num_class = 2)


# ### 2. LOL games datasets

# In[6]:


# Splitting data into training sets and testing sets
X_train, X_test, y_train, y_test = train_test_split(X2,Y2, test_size = 0.2)

#Preprocessing the data between 0 and 1
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
means_init = np.array([X1[Y1 == i].mean(axis=0) for i in range(7)])

try:
    kmeans(X_train, X_test, y_train, y_test, init_means = means_init, component_list = [3,4,5,6,7,8,9,10,11], num_class = 7)
except:
    a=0


# ## (b) Expectation Maximization

# ### 1. NBA games datasets

# In[7]:


# Splitting data into training sets and testing sets
X_train, X_test, y_train, y_test = train_test_split(X1,Y1, test_size = 0.2)

#Preprocessing the data between 0 and 1
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
means_init = np.array([X1[Y1 == i].mean(axis=0) for i in range(2)])

em(X_train, X_test, y_train, y_test, init_means = means_init, component_list = [3,4,5,6,7,8,9,10,11], num_class = 2)


# ### 2. LOL games datasets

# In[8]:


# Splitting data into training sets and testing sets
X_train, X_test, y_train, y_test = train_test_split(X2,Y2, test_size = 0.2)

#Preprocessing the data between 0 and 1
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
means_init = np.array([X1[Y1 == i].mean(axis=0) for i in range(7)])

try:
    em(X_train, X_test, y_train, y_test, init_means = means_init, component_list = [3,4,5,6,7,8,9,10,11], num_class = 7)
except:
    a=0

