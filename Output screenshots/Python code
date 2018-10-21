# -*- coding: utf-8 -*-

#---------------- BEGINNING OF PROJECT -------------------#

"""
Topic : Spooky Author Identification
Problem Type : Classification
Tool : Python

Created by Aparna Adiraju
Created on Sat Oct 20 19:06:00 2018
"""

#---------------- BEGINNING OF PART 1 -------------------#
# 1. Importing required libraries and setting working directory
# 2. Reading data into DataFrames - train, test, positive words, negative words

#importing required libraries
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt; plt.rcdefaults()
import nltk

#get working directory
pwd

#set working directory
os.chdir('C:/Users/aparn/OneDrive/Documents/Learn_Extra/_Kaggle_SpookyAuthor/spooky-author-kaggle')


# importing training data and test data
train = pd.read_csv("C:/Users/aparn/OneDrive/Documents/Learn_Extra/_Kaggle_SpookyAuthor/spooky-author-kaggle/train.csv")
train.head(4)

test = pd.read_csv("C:/Users/aparn/OneDrive/Documents/Learn_Extra/_Kaggle_SpookyAuthor/spooky-author-kaggle/test.csv")
test.head(4)

# importing positive and negative words
pos = open("C:/Users/aparn/OneDrive/Documents/Learn_Extra/_Kaggle_SpookyAuthor/spooky-author-kaggle/positive_words.txt", 'r')
neg = open("C:/Users/aparn/OneDrive/Documents/Learn_Extra/_Kaggle_SpookyAuthor/spooky-author-kaggle/negative_words.txt", 'r')

positive = pos.read().splitlines()  #list
negative = neg.read().splitlines() #list

#---------------- END OF PART 1 -------------------#

#---------------- BEGINNING OF PART 2 -------------------#
# 3. Data Exploration
#    Getting to know more about the data
# 4. Data Visualization 1 - PIE CHART 

# Sample view of training and test data
train.head(4)
test.head(4)

# Observe the count of occurence of authors in training data
authorfreq = train['author'].value_counts()

# Data Visualization of author frequency
authors = authorfreq.index
frequency = authorfreq
colors = ['skyblue', 'yellowgreen', 'lightcoral']
explode = (0.1, 0, 0)  # explodes highest frequency author slice
 
# Plot
plt.pie(frequency, explode=explode, labels=authors, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.title('Author frequency on Training Data')
plt.show()


#---------------- END OF PART 2 -------------------#

#---------------- BEGINNING OF PART 3 -------------------#
# 5. Feature Engineering - train and test set
#    a) Sentiment analysis - score and label  
#       Data Visualization 2    
#    b) adding text length  
#    c) adding comma features
#    d) Bag of words - Natural Language processing

# a) Sentiment analysis
# extracting the desired column "text" from training set and from test set
train_text = train.iloc[:,1].values
train_text[0]

test_text = test.iloc[:,1].values
test_text[0]

#creating new columns for sentiment score and sentiment label in both training and test data
train.loc[:,'sentimentlabel'] = pd.Series("", index=train.index)
train.loc[:,'sentimentscore'] = pd.Series(0, index=train.index)

test.loc[:,'sentimentlabel'] = pd.Series("", index=test.index)
test.loc[:,'sentimentscore'] = pd.Series(0, index=test.index)


#Sentiment analysis function - Function definition is here
posmatch = lambda a, b: [ 1 if x in b else 0 for x in a ]
negmatch = lambda a, c: [ 1 if x in c else 0 for x in a ]
        
def sentimentanalysis( textcolumn, totalrows, positivewords, negativewords, posfunc, negfunc, originaldataset ):
    for i in range(0, totalrows):
        word_list = re.sub('[^a-zA-Z]', ' ', textcolumn[i])
        word_list = word_list.lower()
        word_list = word_list.split()
  
        sentimentscore = np.sum(posmatch(word_list,positive)) - np.sum(negmatch(word_list,negative))
        originaldataset['sentimentscore'][i] = sentimentscore
    
    originaldataset['sentimentlabel'].loc[originaldataset['sentimentscore'] == 0] = "Neutral"
    originaldataset['sentimentlabel'].loc[(originaldataset['sentimentscore'] >=1) & (originaldataset['sentimentscore'] <= 5)] = "Positive"
    originaldataset['sentimentlabel'].loc[(originaldataset['sentimentscore'] <=-1) & (originaldataset['sentimentscore'] >= -5)] = "Negative"
    originaldataset['sentimentlabel'].loc[originaldataset['sentimentscore'] > 5] = "Very Positive"
    originaldataset['sentimentlabel'].loc[originaldataset['sentimentscore'] < -5] = "Very Negative"

# Now you can call sentimentanalysis function
sentimentanalysis(textcolumn = train_text, totalrows = len(train.index), 
                  positivewords = positive, negativewords = negative,
                  posfunc = posmatch, negfunc = negmatch,
                  originaldataset = train)

train['sentimentscore'].describe()
train['sentimentlabel'].value_counts()

sentimentanalysis(textcolumn = test_text, totalrows = len(test.index), 
                  positivewords = positive, negativewords = negative,
                  posfunc = posmatch, negfunc = negmatch,
                  originaldataset = test)

test['sentimentscore'].describe()
test['sentimentlabel'].value_counts()

# Data Visualization of Sentiment analysis - train data
occurence = train['sentimentlabel'].value_counts()
sentimentlabels = occurence.index
y_pos = np.arange(len(sentimentlabels))
 
plt.bar(y_pos, occurence, align='center', alpha=0.5, color = 'green') #plt.bar for vertical bargraph
plt.yticks(y_pos, sentimentlabels)
plt.xlabel('Occurence')
plt.title('Sentiment Analysis on Training Data')
plt.gca().invert_yaxis()
 
plt.show()

# Data Visualization of Sentiment analysis - test data
occurence = test['sentimentlabel'].value_counts()
sentimentlabels = occurence.index
y_pos = np.arange(len(sentimentlabels))
 
plt.bar(y_pos, occurence, align='center', alpha=0.5, color = 'red')  
plt.yticks(y_pos, sentimentlabels)
plt.xlabel('Occurence')
plt.title('Sentiment Analysis on Test Data')
plt.gca().invert_yaxis()
 
plt.show()

# b) Adding Text length feature 
train['textlength'] = train['text'].str.len()
test['textlength'] = test['text'].str.len()

# c) Adding the commas features - number of , : ; .
train['commas'] = train['text'].str.count(",")
train['semicolon'] = train['text'].str.count(";")
train['colon'] = train['text'].str.count(":")
train['dots'] = train['text'].str.count(".")

train.head()


test['commas'] = test['text'].str.count(",")
test['semicolon'] = test['text'].str.count(";")
test['colon'] = test['text'].str.count(":")
test['dots'] = test['text'].str.count(".")

test.head()


# d) Bag of words - Natural Language processing
# Cleaning the texts
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

def Corpusappend(textcolumn, rowlength, corpus):
    for i in range(0, rowlength):
        word_list = re.sub('[^a-zA-Z]', ' ', textcolumn[i])
        word_list = word_list.lower()
        word_list = word_list.split()
        ps = PorterStemmer()
        word_list = [ps.stem(word) for word in word_list if not word in set(stopwords.words('english'))]
        word_list = ' '.join(word_list).lower()
        corpus.append(word_list)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

# del corpus
traincorpus = []
cv = CountVectorizer(max_features = 2500)  # max_features = 15428 in total
Corpusappend(textcolumn = train_text, rowlength = len(train.index), corpus = traincorpus)
cv_X_train = cv.fit_transform(traincorpus).toarray()

testcorpus = []
cv = CountVectorizer(max_features = 2500)
Corpusappend(textcolumn = test_text, rowlength = len(test.index), corpus = testcorpus)
cv_X_test = cv.fit_transform(testcorpus).toarray()


# Binding the Bag of words data to existing training data
output_train = pd.concat([train.reset_index(drop=True), pd.DataFrame(cv_X_train)], axis=1)
output_test = pd.concat([test.reset_index(drop=True), pd.DataFrame(cv_X_test)], axis=1)

#---------------- END OF PART 3 -------------------#

# Saving data into an output file for future reference
train.to_csv("trainhelp.csv")  # sep = '\t'
test.to_csv("testhelp.csv")

#---------------- BEGINNING OF PART 4 -------------------#
# 6. Model building
#    a) creating a model_train and model_test set - stratified split of train data
#    b) Predicting model on model_test data
#    c) Building model - confusion matrix, accuracy
#       Applying K fold cross validation

# Defining our dependent and independent variables
X = output_train.iloc[:, 4:].values
#X = np.delete(X, 1 ,1) # Delete second column from X  (array, number, row = 0 / column = 1)

y = output_train.iloc[:, 2].values

# Stratified split model_train, model_test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# MODEL 1 - Logistic Regression

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
Logisticclassifier = LogisticRegression(random_state = 0)
Logisticclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_logistic = Logisticclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_logistic = confusion_matrix(y_test, y_pred_logistic)
print(cm_logistic)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_logistic = cross_val_score(estimator = Logisticclassifier, X = X_train, y = y_train, cv = 10)
accuracies_logistic.mean()
accuracies_logistic.std()

# MODEL 2 - Naive Bayes

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
NBclassifier = GaussianNB()
NBclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_NB = NBclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_NB = confusion_matrix(y_test, y_pred_NB)
print(cm_NB)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_NB = cross_val_score(estimator = NBclassifier, X = X_train, y = y_train, cv = 10)
accuracies_NB.mean()
accuracies_NB.std()

# MODEL 3 - KNN Classification
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
KNNclassifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
KNNclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_KNN = KNNclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_KNN = confusion_matrix(y_test, y_pred_KNN)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_KNN = cross_val_score(estimator = KNNclassifier, X = X_train, y = y_train, cv = 10)
accuracies_KNN.mean()
accuracies_KNN.std()

# MODEL 4 - Decision Trees
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
DTclassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DTclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = DTclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_DT = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_DT = cross_val_score(estimator = DTclassifier, X = X_train, y = y_train, cv = 10)
accuracies_DT.mean()
accuracies_DT.std()

# MODEL 5 - Random Forest

from sklearn.ensemble import RandomForestClassifier
RFclassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RFclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = RFclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RF = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_RF = cross_val_score(estimator = RFclassifier, X = X_train, y = y_train, cv = 10)
accuracies_RF.mean()
accuracies_RF.std()

# MODEL 6 - XGBoost

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
XGclassifier = XGBClassifier()
XGclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_XG = XGclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_XGB = confusion_matrix(y_test, y_pred_XG)
print(cm_XGB)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_XGB = cross_val_score(estimator = XGclassifier, X = X_train, y = y_train, cv = 10)
accuracies_XGB.mean()
accuracies_XGB.std()

#---------------- END OF PART 4 -------------------#

#---------------- BEGINNING OF PART 5 -------------------#
# 7. Comparing Multiple models - and deciding the best model
# 8. Predicting on the test data set using best model

# Comparing accuracies of different models
comparisonlist = [('Logistic Regression', accuracies_logistic.mean(), accuracies_logistic.std(), accuracies_logistic.mean() + accuracies_logistic.std(), accuracies_logistic.mean() - accuracies_logistic.std()),
                  ('Naive Bayes', accuracies_NB.mean(), accuracies_NB.std(), accuracies_NB.mean() + accuracies_NB.std(), accuracies_NB.mean() - accuracies_NB.std()),
                  ('KNN Classification',accuracies_KNN.mean(), accuracies_KNN.std(), accuracies_KNN.mean() + accuracies_KNN.std(), accuracies_KNN.mean() - accuracies_KNN.std()),
                  ('Decision Trees',accuracies_DT.mean(), accuracies_DT.std(), accuracies_DT.mean() + accuracies_DT.std(), accuracies_DT.mean() - accuracies_DT.std()),
                  ('Random Forest',accuracies_RF.mean(), accuracies_RF.std(), accuracies_RF.mean() + accuracies_RF.std(), accuracies_RF.mean() - accuracies_RF.std()),
                  ('XG Boost',accuracies_XGB.mean(), accuracies_XGB.std(), accuracies_XGB.mean() + accuracies_XGB.std(), accuracies_XGB.mean() - accuracies_XGB.std())]
comparisonlabels = ['Model', 'AccuracyMean', 'AccuracySTD', 'Mean plus STD', 'Mean minus STD']

comparison_df = pd.DataFrame.from_records(comparisonlist, columns = comparisonlabels)

# From the above results, the FINAL MODEL choosen is : LOGISTIC REGRESSION

# Confusion Matrix of the best model
print(cm_logistic)

# Variable importance plot
feature_importance = abs(Logisticclassifier.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

sorted_idx = sorted_idx[0:15]
pos = pos[0:15]

featfig = plt.figure()
featax = featfig.add_subplot(1, 1, 1)
featax.bar(pos, feature_importance[sorted_idx], align='center')
featax.set_xticks(pos)
featax.set_xticklabels(np.array(pd.DataFrame(X_train).columns.values)[sorted_idx], fontsize=8)
featax.set_ylabel('Relative Feature Importance')

plt.tight_layout()  
plt.gca().invert_xaxis()
plt.title('Variable Importance Plot - Top 15')
 
plt.show()
    
# Predicting on the test data set using best model

X_testdata = output_test.iloc[:, 3:].values

y_testdata_pred = Logisticclassifier.predict(X_testdata)
y_testdata_prob = Logisticclassifier.predict_proba(X_testdata)

result = test[['id','text']]
result['author predicted'] = y_testdata_pred
result['EAP probability'] = pd.DataFrame(y_testdata_prob[:, 0])
result['HPL probability'] = pd.DataFrame(y_testdata_prob[:, 1])
result['MWS probability'] = pd.DataFrame(y_testdata_prob[:, 2])

result.head(5)

result.to_csv("output.csv")


#---------------- END OF PART 5 -------------------#

# ================================================= #
#---------------- END OF PROJECT -------------------#
# ================================================= #
