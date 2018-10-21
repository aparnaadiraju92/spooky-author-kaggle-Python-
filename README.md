# Spooky Author Identification - Kaggle
## Using Python   
   
Kaggle Competition link: https://www.kaggle.com/c/spooky-author-identification

This repository helps Data Analytics/Science Enthusiasts to perform feature engineering and predictive analysis on text data provided by Kaggle. I will be using **Python** in this section to complete the exercise. This project has been solved using **R** language too. *https://github.com/aparnaadiraju92/spooky-author-kaggle-R-Studio-*

Data for analysis: https://www.kaggle.com/c/spooky-author-identification/data

# 1. Introduction

What is Predictive analysis?

Out of all the available definitions on the web, the definition that caught my attention was provided by SAS firm in their website. Which states: "Predictive analytics is the use of data, statistical algorithms and machine learning techniques to identify the likelihood of future outcomes based on historical data. The goal is to go beyond knowing what has happened to providing a best assessment of what will happen in the future."

*What is the competition about?*

This competition can be considered as a classic example for predictive modelling and analysis. Competition wants us to train a model based on the given train set, which contains text from various novels written by three writers Edgar Allen Poe (EAP), Mary Shelley (MWS), HP Lovecraft (HPL). Our final goal here is to predict author name for the test dataset which contains only text from the above mentioned authors in random.

Programming Language: Python 

Type of problem : Classification

Algorithm used for training: Logistic Regression, Naive Bayes, KNN Classification, Decision Trees, Random Forest, XGBoost

# 2. Analysis

**Approach to the solution**

Let's divide our analysis into 4 parts:

###### #--------Part-1--------#

1. Importing required libraries and setting working directory

2. Reading data into DataFrames - train, test, positive words, negative words

###### #--------Part-2--------#
3. Data Exploration - Getting to know more about the data
   
4. Data Visualization 1 - **Author occurence frequency in Training data**  *Pie Chart*

![alt text](https://github.com/aparnaadiraju92/spooky-author-kaggle-python-/blob/master/Output%20screenshots/Authorfreq%20-%20training.PNG)

*There are more lines from author Edgar Allen Poe (EAP) in the training data*

###### #--------Part-3--------#
5. Feature Engineering - train and test set

    a) Sentiment analysis and Data Visualization 2 **Score and Label**  *Bar chart (Horizontal and Vertical)*
    
    ![alt text](https://github.com/aparnaadiraju92/spooky-author-kaggle-python-/blob/master/Output%20screenshots/Sentiment%20analysis%20-%20Training.PNG)
    
    ![alt text](https://github.com/aparnaadiraju92/spooky-author-kaggle-python-/blob/master/Output%20screenshots/Sentiment%20analysis%20-%20Test.PNG)
    
    b) adding text length       
    
    c) adding comma features
    
    d) Bag of words - Natural Language processing

###### #--------Part-4--------#
6. Model building

   a) creating a model_train and model_test set - stratified split of train data
   
   b) Building models : ***Logistic Regression, Naive Bayes, KNN Classification, Decision Tree, Random Forest, XGBoost***
   
   c) Applying K-fold cross validation technique for each model

###### #--------Part-5--------#
7. Comparing models 

    ![alt text](https://github.com/aparnaadiraju92/spooky-author-kaggle-python-/blob/master/Output%20screenshots/Models%20comparison.PNG)
   
   *On comparing the various Classification model applied based on the average accuracy from K-fold cross validation, **Logistic Regression** seems to be the best model*
   
8. Predicting values on the test set applying the best model

   ![alt text](https://github.com/aparnaadiraju92/spooky-author-kaggle-python-/blob/master/Output%20screenshots/Sample%20result.PNG)

   
   


