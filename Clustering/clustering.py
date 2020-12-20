#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 22:08:17 2020

@author: rashmisinha
"""
import numpy as np
from functools import cmp_to_key
from matplotlib import pylab as plt
import random

doc_word = np.load("science2k-doc-word.npy")
text_file = open("science2k-titles.txt", "r")
titles = text_file.read().split('\n')

word_doc = np.load("science2k-word-doc.npy")
vocab_file = open("science2k-vocab.txt", "r")
words = vocab_file.read().split('\n')

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    # Function to compute k means by assigning centroids,
    #computing disatance and reassigning centroids
    def fit(self,data):

        self.centroids = {}

        randlist = random.sample(range(data.shape[0]), self.k)
        for i in range(self.k):
            self.centroids[i] = data[randlist[i]]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True
            
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

# Function to compute error for elbow curve    
def geterror(centroid, classification):
        error = 0;
        for i in range(0, len(classification)):
            for j in range(0, len(classification[i])):
                error += (centroid[j] - classification[i][j]) * (centroid[j] - classification[i][j])
        
        return error

# Custom compare to sort according to closeness to a point
def cmp(a, b):
        suma = 0
        for i in range(0, len(a)):
            suma += (centr[i] - a[i]) * (centr[i] - a[i])
            sumb = 0
        for i in range(0, len(b)):
            sumb += (centr[i] - b[i]) * (centr[i] - b[i])
      
        if suma < sumb:
            return 1
        else:
            return -1    
        
# Running k means for doc word dataset        
finans = {}
errorlist = []
for num in range(1, 21):

    clf = K_Means(k = num, max_iter = 1000)
    clf.fit(doc_word) 
    classifications = clf.classifications     
    centroids = clf.centroids    

    global centr


    doc_word_list = doc_word.tolist()
    ans = {}

    error = 0
    for i in centroids:
        error += geterror(centroids[i], classifications[i])
        points = classifications[i]
        cmp_key = cmp_to_key(cmp)
        centr = centroids[i]
        points.sort(key=cmp_key)
        ans[i] = []
        for j in range(0, min(10,  len(points))):
            for p in range(0, len(doc_word_list)):
                if doc_word_list[p] == points[j].tolist():
                    ans[i].append(titles[p])
                    break
    
    errorlist.append(error) 
    finans[num] = ans
    
plt.figure(0)
plt.plot(list(range(1, 21)), errorlist)
plt.ylabel('error')
plt.xlabel('k values')
plt.title('Error versus k values')

#Running k means for word-doc dataset    
finans = {}
errorlist = []
for num in range(1, 21):

    clf = K_Means(k = num, max_iter = 1000)
    clf.fit(word_doc) 
    classifications = clf.classifications     
    centroids = clf.centroids    

    word_doc_list = word_doc.tolist()
    ans = {}

    error = 0
    for i in centroids:
        error += geterror(centroids[i], classifications[i])
        points = classifications[i]
        cmp_key = cmp_to_key(cmp)
        centr = centroids[i]
        points.sort(key=cmp_key)
        ans[i] = []
        for j in range(0, min(10,  len(points))):
            for p in range(0, len(word_doc_list)):
                if doc_word_list[p] == points[j].tolist():
                    ans[i].append(titles[p])
                    break
    
    errorlist.append(error) 
    finans[num] = ans
    
plt.figure(0)
plt.plot(list(range(1, 21)), errorlist)
plt.ylabel('error')
plt.xlabel('k values')
plt.title('Error versus k values')
            