# Spooky Author Identification - Kaggle
## Using Python   
   
Kaggle Competition link: https://www.kaggle.com/c/spooky-author-identification

This repository helps Data Analytics/Science Enthusiasts to perform feature engineering and predictive analysis on text data provided by Kaggle. I will be using **Python** in this section to complete the exercise. This project has been solved using **R** language too. *https://github.com/aparnaadiraju92/spooky-author-kaggle-R-Studio-*

Data for analysis: https://www.kaggle.com/c/spooky-author-identification/data

# 1. Introduction

What is Predictive analysis?

Out of all the available definitions on the web, the definition that caught my attention was provided by SAS firm in their website. Which states: "Predictive analytics is the use of data, statistical algorithms and machine learning techniques to identify the likelihood of future outcomes based on historical data. The goal is to go beyond knowing what has happened to providing a best assessment of what will happen in the future."

*What is the competition about?*

This competition can be considered as a classic example for predictive modelling and analysis. Competition wants us to train a model based on the given train set, which contains text from various novels written by three writers Edgar Allen Poe (EPL), Mary Shelley (MWS), HP Lovecraft (HPL). Our final goal here is to predict author name for the test dataset which contains only text from the above mentioned authors in random.

Programming Language: Python 
                
Algorithm used for training: Logistic Regression, Naive Bayes, XGBoost

# 2. Analysis

**Approach to the solution**

Let's divide our analysis into 4 parts:

###### #--------Part-1--------#

1. Importing required libraries and setting working directory

2. Reading data into DataFrames - train, test, positive words, negative words

###### #--------Part-2--------#
3. Data Exploration - Getting to know more about the data
   
4. Data Visualization 1 - PIE CHART 

###### #--------Part-3--------#
5. Feature Engineering - train and test set

    a) adding text length  

    b) Sentiment analysis - score and label and Data Visualization 2    
    
    c) adding comma features
    
    d) Bag of words - Natural Language processing

###### #--------Part-4--------#
6. Model building

   a) creating a model_train and model_test set - stratified split of train data
   
   b) Building model - confusion matrix, accuracy
   
   c) Predicting model on test data


