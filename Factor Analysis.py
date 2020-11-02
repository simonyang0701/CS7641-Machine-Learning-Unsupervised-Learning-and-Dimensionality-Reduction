#!/usr/bin/env python
# coding: utf-8

# # CS 7641 Assignment 3

# ## Unsupervised Learning and Dimensionality Reduction

# Name: Tianyu Yang<br>
# GTid: 903645962<br>
# Date: 2020/9/20<br>

# In this assignment, I will implement dimensionality reduction algorithms(Factor Analysis) with two datasets, NBA games and LOL games. These two datasets are what I used in the first assignment. I will build up the model for algorithms. Accuracy/Variance, reconstruction error will be plotted to evaluate the performance of dimensionality reduction. Besides, comparing with clustering algorithms, AIC/BIC, performance evaluation, per sample average log likelihood, variance explained figure will be plotted as referred.

# ## Import packages

# In[1]:


get_ipython().system('pip install cluster_func')
# Importing useful packages
import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA
from sklearn import random_projection
from sklearn.decomposition import FactorAnalysis
from cluster_func import kmeans
from cluster_func import em
import scipy

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

X1 = np.array(data1.values[:,1:-1],dtype='float')
Y1 = np.array(data1.values[:,0],dtype='float')
data1.describe(include='all')


# ### 2. LOL games data

# We will analyze the lol matches data There are two teams called 1 and 2. From six aspects including first blood, first tower, first inhibitor, first Baron, first Dragon, first RiftHerald to analyze the winner of each game.

# In[4]:


data2 = lol_games[['winner','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald']]
data2 = data2.astype(np.float64, copy=False)

# We only used 965 data for testing which is the same as data1
data2 = data2.head(965)
#X2 = data2.values[:,1:-1]
#Y2 = data2.values[:,0]

X2 = np.array(data1.values[:,1:-1],dtype='float')
Y2 = np.array(data1.values[:,0],dtype='float')
data2.describe(include='all')


# ## Factor Analysis dimensionality reduction algorithm

# ### 1. NBA games datasets

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X1,Y1, test_size = 0.2)


# In[6]:


# Dimensionality reduction Factor Analysis


print("Starting FA")
print("Dimensionality reduction")

nn = MLPClassifier(hidden_layer_sizes=(30,), solver='adam', activation='logistic', learning_rate_init=0.05, 
                                max_iter=1000, random_state=100)
fa = FactorAnalysis(max_iter = 100)

pipe = Pipeline(steps=[('fa', fa), ('neural networks', nn)])

# Plot the PCA spectrum
fa.fit(X1)

fig, ax = plt.subplots()
print(list(range(1,X1.shape[1])),fa.noise_variance_)

ax.bar(np.arange(X1.shape[1]), fa.noise_variance_, linewidth=2, color = 'blue')
plt.axis('tight')
plt.xlabel('n_components')
ax.set_ylabel('noise variance')

#Checking the accuracy for taking all combination of components
n_components = range(1, X1.shape[1])

gridSearch = GridSearchCV(pipe, dict(fa__n_components=n_components), cv = 3)
gridSearch.fit(X1, Y1)
results = gridSearch.cv_results_
ax1 = ax.twinx()

#Plotting the accuracies and best component
ax1.plot(results['mean_test_score'], linewidth = 2, color = 'red')
ax1.set_ylabel('Mean Cross Validation Accuracy')
ax1.axvline(gridSearch.best_estimator_.named_steps['fa'].n_components, linestyle=':', label='n_components chosen', linewidth = 2)

plt.legend(prop=dict(size=12))
plt.title('Accuracy/Noise Variance for FA (best n_components=  %d)'%gridSearch.best_estimator_.named_steps['fa'].n_components )
plt.show()

#Reducing the dimensions with optimal number of components
fa_new = FactorAnalysis(n_components = gridSearch.best_estimator_.named_steps['fa'].n_components, max_iter = 100)
fa_new.fit(X_train)
X_train_transformed = fa_new.transform(X_train)
X_test_transformed = fa_new.transform(X_test)


# In[7]:


#Reconstruction Error

print("Calculating Reconstruction Error")

def inverse_transform_fa(fa, X_transformed, X_train):

	return X_transformed.dot(fa.components_) + np.mean(X_train, axis = 0)

reconstruction_error = []

for comp in n_components:

	fa = FactorAnalysis(n_components = comp, max_iter = 100)
	X_transformed = fa.fit_transform(X_train)
	X_projected = inverse_transform_fa(fa, X_transformed, X_train)
	reconstruction_error.append(((X_train - X_projected) ** 2).mean())

	if(comp == gridSearch.best_estimator_.named_steps['fa'].n_components):
		chosen_error = ((X_train - X_projected) ** 2).mean()

fig2,ax2 = plt.subplots()
ax2.plot(n_components, reconstruction_error, linewidth= 2)
ax2.axvline(gridSearch.best_estimator_.named_steps['fa'].n_components, linestyle=':', label='n_components chosen', linewidth = 2)
plt.axis('tight')
plt.xlabel('Number of components')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction error for n_components chosen %f '%chosen_error)
plt.show()


# In[8]:


#Clustering after dimensionality reduction

print("Clustering FA")

#Reducing the dimensions with optimal number of components
fa_new = FactorAnalysis(n_components = gridSearch.best_estimator_.named_steps['fa'].n_components, max_iter = 100)
fa_new.fit(X1)
X_transformed = fa_new.transform(X1)


means_init = np.array([X_transformed[Y1 == i].mean(axis=0) for i in range(2)])

#clustering experiments
print("Expected Maximization")
component_list, array_aic, array_bic, array_homo_1, array_comp_1, array_sil_1, array_avg_log = em(X_train_transformed, X_test_transformed, y_train, y_test, init_means = means_init, component_list = [3,4,5,6,7,8,9,10,11], num_class = 2, toshow = 0)

print("KMeans")
component_list, array_homo_2, array_comp_2, array_sil_2, array_var = kmeans(X_train_transformed, X_test_transformed, y_train, y_test, init_means = means_init, component_list = [3,4,5,6,7,8,9,10,11], num_class = 2, toshow = 0)


# In[9]:


#Writing data to file
component_list = np.array(component_list).reshape(-1,1)
array_aic = np.array(array_aic).reshape(-1,1)
array_bic = np.array(array_bic).reshape(-1,1)
array_homo_1 = np.array(array_homo_1).reshape(-1,1)
array_comp_1 = np.array(array_comp_1).reshape(-1,1)
array_sil_1 = np.array(array_sil_1).reshape(-1,1)
array_avg_log = np.array(array_avg_log).reshape(-1,1)
array_homo_2 = np.array(array_homo_2).reshape(-1,1)
array_comp_2 = np.array(array_comp_2).reshape(-1,1)
array_sil_2 = np.array(array_sil_2).reshape(-1,1)
array_var = np.array(array_var).reshape(-1,1)

reconstruction_error = np.array(reconstruction_error).reshape(-1,1)

data_em_fa_cancer = np.concatenate((component_list, array_aic, array_bic, array_homo_1, array_comp_1, array_sil_1, array_avg_log), axis =1)

data_km_fa_cancer = np.concatenate((component_list, array_homo_2, array_sil_2, array_var), axis =1)

reconstruction_error_fa_cancer = np.concatenate((np.arange(1,X1.shape[1]).reshape(-1,1), reconstruction_error), axis = 1)

file = './outputs/data_em_fa_nba.csv'
with open(file, 'w', newline = '') as output:
	writer = csv.writer(output, delimiter=',')
	writer.writerows(data_em_fa_cancer)

file = './outputs/data_km_fa_nba.csv'
with open(file, 'w', newline = '') as output:
	writer = csv.writer(output, delimiter=',')
	writer.writerows(data_km_fa_cancer)

file = './outputs/reconstruction_error_fa_nba.csv'
with open(file, 'w', newline = '') as output:
	writer = csv.writer(output, delimiter=',')
	writer.writerows(reconstruction_error_fa_cancer)


# ### 2. LOL games datasets

# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X2,Y2, test_size = 0.2)


# In[11]:


# Dimensionality reduction Factor Analysis


print("Starting FA")
print("Dimensionality reduction")

nn = MLPClassifier(hidden_layer_sizes=(30,), solver='adam', activation='logistic', learning_rate_init=0.05, 
                                max_iter=1000, random_state=100)
fa = FactorAnalysis(max_iter = 100)

pipe = Pipeline(steps=[('fa', fa), ('neural networks', nn)])

# Plot the PCA spectrum
fa.fit(X2)

fig, ax = plt.subplots()
print(list(range(1,X2.shape[1])),fa.noise_variance_)

ax.bar(np.arange(X2.shape[1]), fa.noise_variance_, linewidth=2, color = 'blue')
plt.axis('tight')
plt.xlabel('n_components')
ax.set_ylabel('noise variance')

#Checking the accuracy for taking all combination of components
n_components = range(1, X2.shape[1])

gridSearch = GridSearchCV(pipe, dict(fa__n_components=n_components), cv = 3)
gridSearch.fit(X2, Y2)
results = gridSearch.cv_results_
ax1 = ax.twinx()

#Plotting the accuracies and best component
ax1.plot(results['mean_test_score'], linewidth = 2, color = 'red')
ax1.set_ylabel('Mean Cross Validation Accuracy')
ax1.axvline(gridSearch.best_estimator_.named_steps['fa'].n_components, linestyle=':', label='n_components chosen', linewidth = 2)

plt.legend(prop=dict(size=12))
plt.title('Accuracy/Noise Variance for FA (best n_components=  %d)'%gridSearch.best_estimator_.named_steps['fa'].n_components )
plt.show()

#Reducing the dimensions with optimal number of components
fa_new = FactorAnalysis(n_components = gridSearch.best_estimator_.named_steps['fa'].n_components, max_iter = 100)
fa_new.fit(X_train)
X_train_transformed = fa_new.transform(X_train)
X_test_transformed = fa_new.transform(X_test)


# In[12]:


#Reconstruction Error

print("Calculating Reconstruction Error")

def inverse_transform_fa(fa, X_transformed, X_train):

	return X_transformed.dot(fa.components_) + np.mean(X_train, axis = 0)

reconstruction_error = []

for comp in n_components:

	fa = FactorAnalysis(n_components = comp, max_iter = 100)
	X_transformed = fa.fit_transform(X_train)
	X_projected = inverse_transform_fa(fa, X_transformed, X_train)
	reconstruction_error.append(((X_train - X_projected) ** 2).mean())

	if(comp == gridSearch.best_estimator_.named_steps['fa'].n_components):
		chosen_error = ((X_train - X_projected) ** 2).mean()

fig2,ax2 = plt.subplots()
ax2.plot(n_components, reconstruction_error, linewidth= 2)
ax2.axvline(gridSearch.best_estimator_.named_steps['fa'].n_components, linestyle=':', label='n_components chosen', linewidth = 2)
plt.axis('tight')
plt.xlabel('Number of components')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction error for n_components chosen %f '%chosen_error)
plt.show()


# In[13]:


#Clustering after dimensionality reduction

print("Clustering FA")

#Reducing the dimensions with optimal number of components
fa_new = FactorAnalysis(n_components = gridSearch.best_estimator_.named_steps['fa'].n_components, max_iter = 100)
fa_new.fit(X2)
X_transformed = fa_new.transform(X2)


means_init = np.array([X_transformed[Y2 == i].mean(axis=0) for i in range(2)])

#clustering experiments
print("Expected Maximization")
component_list, array_aic, array_bic, array_homo_1, array_comp_1, array_sil_1, array_avg_log = em(X_train_transformed, X_test_transformed, y_train, y_test, init_means = means_init, component_list = [3,4,5,6,7,8,9,10,11], num_class = 2, toshow = 0)

print("KMeans")
component_list, array_homo_2, array_comp_2, array_sil_2, array_var = kmeans(X_train_transformed, X_test_transformed, y_train, y_test, init_means = means_init, component_list = [3,4,5,6,7,8,9,10,11], num_class = 2, toshow = 0)


# In[14]:


#Writing data to file
component_list = np.array(component_list).reshape(-1,1)
array_aic = np.array(array_aic).reshape(-1,1)
array_bic = np.array(array_bic).reshape(-1,1)
array_homo_1 = np.array(array_homo_1).reshape(-1,1)
array_comp_1 = np.array(array_comp_1).reshape(-1,1)
array_sil_1 = np.array(array_sil_1).reshape(-1,1)
array_avg_log = np.array(array_avg_log).reshape(-1,1)
array_homo_2 = np.array(array_homo_2).reshape(-1,1)
array_comp_2 = np.array(array_comp_2).reshape(-1,1)
array_sil_2 = np.array(array_sil_2).reshape(-1,1)
array_var = np.array(array_var).reshape(-1,1)

reconstruction_error = np.array(reconstruction_error).reshape(-1,1)

data_em_fa_cancer = np.concatenate((component_list, array_aic, array_bic, array_homo_1, array_comp_1, array_sil_1, array_avg_log), axis =1)

data_km_fa_cancer = np.concatenate((component_list, array_homo_2, array_sil_2, array_var), axis =1)

reconstruction_error_fa_cancer = np.concatenate((np.arange(1,X2.shape[1]).reshape(-1,1), reconstruction_error), axis = 1)

file = './outputs/data_em_fa_lol.csv'
with open(file, 'w', newline = '') as output:
	writer = csv.writer(output, delimiter=',')
	writer.writerows(data_em_fa_cancer)

file = './outputs/data_km_fa_lol.csv'
with open(file, 'w', newline = '') as output:
	writer = csv.writer(output, delimiter=',')
	writer.writerows(data_km_fa_cancer)

file = './outputs/reconstruction_error_fa_lol.csv'
with open(file, 'w', newline = '') as output:
	writer = csv.writer(output, delimiter=',')
	writer.writerows(reconstruction_error_fa_cancer)

