#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:04:42 2020

@author: rashmisinha
"""

import numpy as np
import imageio

from matplotlib import pylab as plt
import matplotlib.cm as cm
from numpy import linalg as LA
from sklearn.linear_model import LogisticRegression

#Loading train data
train_labels, train_data = [], []
for line in open('./faces/train.txt'):
    im = imageio.imread(line.strip().split()[0])
    train_data.append(im.reshape(2500,))
    train_labels.append(line.strip().split()[1])
    X = np.array(train_data, dtype=float)
    train_labels_arr = np.array(train_labels, dtype=int)

#displaying an image from train data
print(X.shape, train_labels_arr.shape)
plt.imshow(X[10, :].reshape(50,50), cmap = cm.Greys_r)
plt.axis('off')
plt.show()

#Loading test data
test_labels, test_data = [], []
for line in open('./faces/test.txt'):
    im = imageio.imread(line.strip().split()[0])
    test_data.append(im.reshape(2500,))
    test_labels.append(line.strip().split()[1])
    X_test = np.array(test_data, dtype=float)
    test_labels_arr = np.array(test_labels, dtype=int)

#Displaying an image form test data
print(X_test.shape, test_labels_arr.shape)
plt.imshow(X_test[10, :].reshape(50,50), cmap = cm.Greys_r)
plt.axis('off')
plt.show()

#Computing average face
average_face = X.mean(axis=0)

#Displaying average face
plt.imshow(average_face.reshape(50,50), cmap = cm.Greys_r)
plt.axis('off')
plt.show()

#Computer mean subtraction on train set
for row in range(0, X.shape[0]):
    X[row] = np.subtract(X[row], average_face)

#Computer mean subtraction on test set
for row in range(0, X_test.shape[0]):
    X_test[row] = np.subtract(X_test[row], average_face)

#Displaying mean subtraction on train set 
plt.imshow(X[10, :].reshape(50,50), cmap = cm.Greys_r)
plt.axis('off')
plt.show()

#Displaying mean subtraction on test set 
plt.imshow(X_test[10, :].reshape(50,50), cmap = cm.Greys_r)
plt.axis('off')
plt.show()

# Computing eigen faces
mul = np.transpose(X) @ X;
v, s, vh = np.linalg.svd(mul)

eigen_vector_transpose = np.transpose(v)

for i in range(0,10):
    plt.imshow(eigen_vector_transpose[i, :].reshape(50,50), cmap = cm.Greys_r)
    plt.axis('off')
    plt.show()
    
#Writing function to get feature matrix
def feature_matrix(r, X, X_test, V):
    V_transpose = np.transpose(V)
    V_transpose = V_transpose[:r,:]
    
    F = X @ np.transpose(V_transpose)
    F_test = X_test @ np.transpose(V_transpose)
    
    return F, F_test

# Computing score for r = 10
f,f_test = feature_matrix(10, X, X_test, v)

clf = LogisticRegression(random_state=0, max_iter=1000, multi_class = 'ovr').fit(f, train_labels_arr)
F_pred = clf.predict(f_test)

score = clf.score(f_test, test_labels_arr)

# Computing scores for r = 1 to 200
accuracy = []
for r in range(1,201):
    f,f_test = feature_matrix(r, X, X_test, v)

    clf = LogisticRegression(random_state=0, max_iter=10000, multi_class = 'ovr').fit(f, train_labels_arr)
    F_pred = clf.predict(f_test)

    score = clf.score(f_test, test_labels_arr)
    accuracy.append(score)

#Displaying plot for Accuracy versus r values    
plt.figure(0)
plt.plot(list(range(1, 201)), accuracy)
plt.xlabel('r values')
plt.ylabel('accuracy')
plt.title('Accuracy versus R Values')
    