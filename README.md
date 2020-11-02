# CS 7641 Assignment 3: Unsupervised Learning and Dimensionality Reduction
## Author
Name: Tianyu Yang<br>
GTid: 903645962<br>
Date: 2020/10/31<br>

## Introduction
In this unsupervised learning and dimensionality reduction project, two clustering algorithms (k-means clustering and expectation maximization) and four dimensionality reduction algorithms (PCA, ICA, Randomized Projections and Factor Analysis) will be implemented. Also, this program will also demonstrate the improvement of the performance after using dimensionality reduction methods. The datasets I used is from the assignment 1 which is the NBA games and LOL games.The datasets I used is downloaded from Kaggle<br>

NBA games: https://www.kaggle.com/nathanlauga/nba-games<br>
LOL games: https://www.kaggle.com/datasnaek/league-of-legends?select=games.csv<br>

This project link on Github: https://github.com/simonyang0701/CS7641-Machine-Learning-Unsupervised-Learning-and-Dimensionality-Reduction.git<br>


## Getting Started & Prerequisites
To test the code, you need to make sure that your python 3.6 is in recent update and the following packages have already been install:
pandas, numpy, scikit-learn, matplotlib, scipy

## Running the models
Recommendation Option: Work with the iPython notebook (.ipnyb) using Jupyter or a similar environment. Use "Run ALL" in Cell to run the code. Before running the code, make sure that you have already change the path into your current working directory
Another Option: Run the python script (.py) after changing the directory into where you saved the two datasets
Other Option (view only): Feel free to open up the (.html) file to see the output of program

The two Clustering Algorithms are divided into four parts:
1. Importing useful packages
2. Loading and cleaning datasets
3. Building up the model and training the datasets
4. Plotting the result in figure

The four dimensionality reduction algorithms codes are divided into six parts:
1. Importing useful packages
2. Loading and cleaning datasets: load and clean the datasets
3. Running the dimensionality reudction algorithms and plot hte figures
4. Getting the reconstruction errors
5. Using clustering algorithms after dimensionality reduction
6.Writing the results into files
