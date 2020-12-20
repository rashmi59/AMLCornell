#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:30:25 2020

@author: rashmisinha
"""
import pandas as pd
import numpy as np
import string
import nltk
import random
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import matplotlib.pyplot as plt 

#reading training data
train_data = pd.read_csv("train.csv")

#number of training data
train_count = train_data.shape[0]

#percentage of tweets real disasters
disaster_percent = (train_data[train_data.target == 1].shape[0] * 100)/train_count;
not_disaster_percent = (train_data[train_data.target == 0].shape[0] * 100)/train_count;

#reading test data
test_data = pd.read_csv("test.csv")

#number of training data
test_count = test_data.shape[0]

#preprocessing steps
#to lower case
train_data['text'] = train_data['text'].str.lower()

# Use English stemmer.
def stem_sentences(sentence):
    stemmer = PorterStemmer()
    tokens = sentence.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

train_data['text'] = train_data['text'].apply(stem_sentences)
test_data['text'] = test_data['text'].apply(stem_sentences)

#remove punctuation
def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))

train_data['text'] = train_data['text'].apply(remove_punctuation)
test_data['text'] = test_data['text'].apply(remove_punctuation)

#remove stop words
def remove_stopwords(sentence):
    word_list = sentence.split()
    filtered_words = [word for word in word_list if word not in stopwords.words('english')]
    return ' '.join(filtered_words)

train_data['text'] = train_data['text'].apply(remove_stopwords)
test_data['text'] = test_data['text'].apply(remove_stopwords)

# vectorize the training 

count_vect=CountVectorizer(binary=True, min_df= 10)
count_vect.fit(train_data['text'])
train_data_vec = pd.DataFrame(count_vect.transform(train_data['text']).todense(),
                              columns = [str(col) + '_word' for col in count_vect.get_feature_names()])
test_data_vec = pd.DataFrame(count_vect.transform(test_data['text']).todense(),
                             columns = [str(col) + '_word' for col in count_vect.get_feature_names()])

#uncomment below for extended column
#train_data = train_data.replace(np.nan, '', regex=True)
#train_data_vec['location'] = LabelEncoder().fit_transform(train_data['location'])

#test_data = test_data.replace(np.nan, '', regex=True)
#test_data_vec['location'] = LabelEncoder().fit_transform(test_data['location'])

train_data_vec['target'] = train_data['target']

#randomly split data into train set and dev set
train_set = train_data_vec.sample(frac = 0.7, random_state = 1)
dev_set = train_data_vec.drop(train_set.index)

#create input training and dev set
train_set_x = train_set.loc[:, train_set.columns != 'target']
dev_set_x = dev_set.loc[:, dev_set.columns != 'target']

#create output for training and dev set
train_set_y = train_set['target']
dev_set_y = dev_set['target']

#naive bayes implementation
X_train = train_set_x.to_numpy()
y_train = train_set_y

n=X_train.shape[0]
# size of the dataset
d=X_train.shape[1]
# number of features in our dataset
K=2# number of clases

# these are the shapes of the parameters
psis=np.zeros([K,d])
phis=np.zeros([K])

# we now compute the parameters
for k in range(K):
    X_k=X_train[y_train==k]
    psis[k]=np.mean(X_k, axis=0)
    phis[k]=X_k.shape[0]/float(n)

def nb_predictions(x, psis, phis):
    """This returns class assignments and scores under the NB model.
    We compute \arg\max_y p(y|x) as \arg\max_y p(x|y)p(y)"""
    # adjust shapes
    n, d = x.shape
    x = np.reshape(x, (1, n, d))
    psis = np.reshape(psis, (K, 1, d))
    
    # clip probabilities to avoid log(0)
    psis = psis.clip(1e-14, 1-1e-14)

    # compute log-probabilities
    logpy = np.log(phis).reshape([K,1])
    logpxy = x * np.log(psis) + (1-x) * np.log(1-psis)
    logpyx = logpxy.sum(axis=2) + logpy
    return logpyx.argmax(axis=0).flatten(), logpyx.reshape([K,n])

idx, logpyx = nb_predictions(X_train, psis, phis)

#get predicted values for dev set
dev_set_pred, logpyx_new=nb_predictions(dev_set_x.to_numpy(), psis, phis)

#compute f1 score
NaiveBayesF1 = f1_score(dev_set_y, dev_set_pred, average='macro')

#get predicted values for test set
test_pred, logpyx_new=nb_predictions(test_data_vec.to_numpy(), psis, phis)
test_pred_df = pd.DataFrame(test_pred, columns=['target'])
test_pred_df.insert(0,'id',test_data['id'])
test_pred_df.to_csv(r'NaiveBayes1Gram.csv')

#logistic regression
clf = LogisticRegression(random_state=0, max_iter=1000).fit(train_set_x, train_set_y)

#get predicted values for dev set
dev_set_pred = clf.predict(dev_set_x)

#compute f1 score
LogisticF1 = f1_score(dev_set_y, dev_set_pred, average='macro')

#get predicted values for test set
test_pred = clf.predict(test_data_vec)
test_pred_df = pd.DataFrame(test_pred, columns=['target'])
test_pred_df.insert(0,'id',test_data['id'])
test_pred_df.to_csv(r'Logistic1Gram.csv')

#Find most important words
logistic_tokens = pd.DataFrame(
    data=clf.coef_[0],
    index=count_vect.get_feature_names(),
    columns=['coefficient']
).sort_values(by = 'coefficient', ascending=False).to_csv(r'Logistic1GramCoefficient.csv')

#uncomment for appended columns
'''k = count_vect.get_feature_names()
k.append(['location'])
zippedlist = zip(clf.coef_[0], k)
sorted_list1 = [element for _, element in sorted(zippedlist, reverse = True)]
pd.DataFrame(sorted_list1, columns=['coefficient']).to_csv(r'Logistic1GramExtendedCoefficient.csv')

coef_list_logistic = clf.coef_[0].copy()
sorted(coef_list_logistic, reverse = True)'''

#Linear SVM Prediction
C_vec = [0.01, 0.1, 1.0, 10.0, 100.0]
F1_scores = []
mx_f1 = -1
best_c = 0.01
for c in C_vec:
    clf = svm.LinearSVC(C= c, max_iter=10000000).fit(train_set_x, train_set_y)
    #get predicted values for dev set
    dev_set_pred = clf.predict(dev_set_x)
    #compute f1 score
    f1 = f1_score(dev_set_y, dev_set_pred, average='macro')
    F1_scores.append(f1)
    
    if f1 > mx_f1:
        mx_f1 = f1
        best_clf = clf
        best_c = c

#Plot parameter versus f1 scores
plt.figure(0)
plt.plot(C_vec, F1_scores)
plt.xlabel('Parameters')
plt.ylabel('F1 scores')
plt.title('Linear SVM for 1-gram')

#get predicted values for test set
test_pred = best_clf.predict(test_data_vec)
test_pred_df = pd.DataFrame(test_pred, columns=['target'])
test_pred_df.insert(0,'id',test_data['id'])
test_pred_df.to_csv(r'LinearSVM1Gram.csv')

#Find most important words
linearSVM_tokens = pd.DataFrame(
    data=best_clf.coef_[0],
    index=count_vect.get_feature_names(),
    columns=['coefficient']
).sort_values(by = 'coefficient', ascending=False).to_csv(r'Linear1GramCoefficient.csv')

#uncomment for appended columns
'''k = count_vect.get_feature_names()
k.append(['location'])
zippedlist = zip(clf.coef_[0], k)
sorted_list1 = [element for _, element in sorted(zippedlist, reverse = True)]
pd.DataFrame(sorted_list1, columns=['coefficient']).to_csv(r'Linear1GramExtendedCoefficient.csv')

linearsvm = clf.coef_[0].copy()
sorted(linearsvm, reverse = True)'''

#Non linear SVM
C_vec = [0.01, 0.1, 1.0, 10.0, 100.0]
F1_scores = []
mx_f1 = -1
best_c = 0.01
for c in C_vec:
    clf = svm.SVC(kernel='rbf', gamma='auto', C=c).fit(train_set_x, train_set_y)
    #get predicted values for dev set
    dev_set_pred = clf.predict(dev_set_x)
    #compute f1 score
    f1 = f1_score(dev_set_y, dev_set_pred, average='macro')
    F1_scores.append(f1)
    
    if f1 > mx_f1:
       mx_f1 = f1
       best_clf = clf
       best_c = c

#Plot parameter versus f1 scores
plt.figure(1)
plt.plot(C_vec, F1_scores)
plt.xlabel('Parameters')
plt.ylabel('F1 scores')
plt.title('SVM for 1-gram')

#get predicted values for test set
test_pred = best_clf.predict(test_data_vec)
test_pred_df = pd.DataFrame(test_pred, columns=['target'])
test_pred_df.insert(0,'id',test_data['id'])
test_pred_df.to_csv(r'SVM1Gram.csv')

#N-Gram Model
count_vect_2gram=CountVectorizer(binary=True, ngram_range=(1,2), min_df= 10)
count_vect_2gram.fit(train_data['text'])

number_2_grams = [s for s in count_vect_2gram.get_feature_names() if ' ' in s]
number_1_grams = [s for s in  count_vect_2gram.get_feature_names() if s not in number_2_grams]
sample_2gram = random.sample(number_2_grams, 10)

train_data_vec_2gram = pd.DataFrame(count_vect_2gram.transform(train_data['text']).todense(),
                                    columns = [str(col) + '_word' for col in count_vect_2gram.get_feature_names()])
test_data_vec = pd.DataFrame(count_vect_2gram.transform(test_data['text']).todense(),
                             columns = [str(col) + '_word' for col in count_vect_2gram.get_feature_names()])

train_data_vec_2gram['target'] = train_data['target']
train_set = train_data_vec_2gram.sample(frac = 0.7, random_state = 1)
dev_set = train_data_vec_2gram.drop(train_set.index)

train_set_x = train_set.loc[:, train_set.columns != 'target']
dev_set_x = dev_set.loc[:, dev_set.columns != 'target']

train_set_y = train_set['target']
dev_set_y = dev_set['target']

#naive bayes implementation
X_train = train_set_x.to_numpy()
y_train = train_set_y

n=X_train.shape[0]
# size of the dataset
d=X_train.shape[1]
# number of features in our dataset
K=2# number of clases

# these are the shapes of the parameters
psis=np.zeros([K,d])
phis=np.zeros([K])

# we now compute the parameters
for k in range(K):
    X_k=X_train[y_train==k]
    psis[k]=np.mean(X_k, axis=0)
    phis[k]=X_k.shape[0]/float(n)

idx, logpyx = nb_predictions(X_train, psis, phis)

#compute predicted values for dev set
dev_set_pred, logpyx_new=nb_predictions(dev_set_x.to_numpy(), psis, phis)

#compute f1 score
NaiveBayesF1_2gram = f1_score(dev_set_y, dev_set_pred, average='macro')

#compute predicted values for test set
test_pred, logpyx_new=nb_predictions(test_data_vec.to_numpy(), psis, phis)
test_pred_df = pd.DataFrame(test_pred, columns=['target'])
test_pred_df.insert(0,'id',test_data['id'])
test_pred_df.to_csv(r'NaiveBayes2Gram.csv')

#logistic regression
clf = LogisticRegression(random_state=0).fit(train_set_x, train_set_y)

#compute predicted values for dev set
dev_set_pred = clf.predict(dev_set_x)

#compute f1 score
LogisticF1_2gram = f1_score(dev_set_y, dev_set_pred, average='macro')

#compute top most important words
logistic_tokens = pd.DataFrame(
    data=clf.coef_[0],
    index=count_vect_2gram.get_feature_names(),
    columns=['coefficient']
).sort_values(by = 'coefficient', ascending=False).to_csv(r'Logistic2GramCoefficient.csv')

#get predicted values for test set
test_pred = clf.predict(test_data_vec)
test_pred_df = pd.DataFrame(test_pred, columns=['target'])
test_pred_df.insert(0,'id',test_data['id'])
test_pred_df.to_csv(r'Logistic2Gram.csv')

#Linear SVM Prediction
C_vec = [0.01, 0.1, 1.0, 10.0, 100.0]
F1_scores = []
mx_f1 = -1
best_c = 0.01
for c in C_vec:
    clf = svm.LinearSVC(C= c, max_iter=100000).fit(train_set_x, train_set_y)
    #compute predicted values for dev set
    dev_set_pred = clf.predict(dev_set_x)
    #compute f1 score
    f1 = f1_score(dev_set_y, dev_set_pred, average='macro')
    F1_scores.append(f1)
    
    if f1 > mx_f1:
       mx_f1 = f1
       best_clf = clf
       best_c = c

#plot parameter versus f1 scores
plt.figure(2)
plt.plot(C_vec, F1_scores)
plt.xlabel('Parameters')
plt.ylabel('F1 scores')
plt.title('Linear SVM for 2-gram')

LinearSVMF1 = mx_f1
LinearSVMC = best_c

#predict values for test set
test_pred = best_clf.predict(test_data_vec)
test_pred_df = pd.DataFrame(test_pred, columns=['target'])
test_pred_df.insert(0,'id',test_data['id'])
test_pred_df.to_csv(r'LinearSVM2Gram.csv')

#compute top important words
linearSVM_tokens = pd.DataFrame(
    data=best_clf.coef_[0],
    index=count_vect_2gram.get_feature_names(),
    columns=['coefficient']
).sort_values(by = 'coefficient', ascending=False).to_csv(r'Linear2GramCoefficient.csv')

#Non linear SVM
C_vec = [0.01, 0.1, 1.0, 10.0, 100.0]
F1_scores = []
mx_f1 = -1
best_c = 0.01
for c in C_vec:
    clf = svm.SVC(kernel='rbf', gamma='auto', C=c).fit(train_set_x, train_set_y)
    #compute predicted values for dev set
    dev_set_pred = clf.predict(dev_set_x)
    #compute f1 score
    f1 = f1_score(dev_set_y, dev_set_pred, average='macro')
    F1_scores.append(f1)
    
    if f1 > mx_f1:
       mx_f1 = f1
       best_clf = clf
       best_c = c
      
#plot parameters with f1 scores
plt.figure(3)
plt.plot(C_vec, F1_scores)
plt.xlabel('Parameters')
plt.ylabel('F1 scores')
plt.title('SVM for 2-gram')

svmF1 = mx_f1
svmC = best_c

#compute predicted values with test set
test_pred = best_clf.predict(test_data_vec)
test_pred_df = pd.DataFrame(test_pred, columns=['target'])
test_pred_df.insert(0,'id',test_data['id'])
test_pred_df.to_csv(r'SVM2Gram.csv')
