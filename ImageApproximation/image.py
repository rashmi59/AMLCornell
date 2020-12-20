#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:22:01 2020

@author: rashmisinha
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn import tree
from sklearn.tree import export_graphviz
import pydot
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
from sklearn.neighbors import KNeighborsRegressor

#reading the input image
img=mpimg.imread('monalisa.jpg')
imgplot = plt.imshow(img)

#choosing random coordinates
r_x_coords = np.random.randint(img.shape[0], size=(5000, 1))
r_y_coords = np.random.randint(img.shape[1], size=(5000, 1))

r_coords = np.concatenate((r_x_coords, r_y_coords), axis=1)

#splitting output into 3 different channels
y_r = []
y_g = []
y_b = []
for coord in r_coords:
    y_r.append(img[coord[0], coord[1],0])
    y_g.append(img[coord[0], coord[1],1])
    y_b.append(img[coord[0], coord[1],2])


#Random Forest Regressor
regressor_r = RandomForestRegressor(n_estimators=1, max_depth = 7, ccp_alpha = 0.0, random_state=0)
regressor_g = RandomForestRegressor(n_estimators=1, max_depth = 7, ccp_alpha = 0.0, random_state=0)
regressor_b = RandomForestRegressor(n_estimators=1, max_depth = 7, ccp_alpha = 0.0, random_state=0)

'''
#KNN Regressor - Comment this block t run Random Forest Regressor
regressor_r = KNeighborsRegressor(n_neighbors = 1)
regressor_g = KNeighborsRegressor(n_neighbors = 1)
regressor_b = KNeighborsRegressor(n_neighbors = 1)
'''

#fititng on training data
regressor_r.fit(r_coords, y_r)
regressor_g.fit(r_coords, y_g)
regressor_b.fit(r_coords, y_b)

coordinate = [] 
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        coordinate.append([i,j])

#predicting the outputs
y_r_pred = regressor_r.predict(coordinate)
y_g_pred = regressor_g.predict(coordinate)
y_b_pred = regressor_b.predict(coordinate)

#converting the output to 2D, as image is int 2D
y_r_pred_2d = np.reshape(y_r_pred, (img.shape[0], img.shape[1]))
y_g_pred_2d = np.reshape(y_g_pred, (img.shape[0], img.shape[1]))
y_b_pred_2d = np.reshape(y_b_pred, (img.shape[0], img.shape[1]))

#combining the 3 planes into a color image
rgbArray = np.zeros((img.shape[0],img.shape[1],3), 'uint8')
rgbArray[..., 0] = y_r_pred_2d
rgbArray[..., 1] = y_g_pred_2d
rgbArray[..., 2] = y_b_pred_2d
final_img = Image.fromarray(rgbArray)
plt.imshow(final_img)
plt.axis('off')
final_img.save('myimg.jpg')

# extract a decision tree in the random forest
tree_r = regressor_r.estimators_[0]

#saving tree as dot file
export_graphviz(tree_r, out_file = 'tree.dot', rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')

#plotting the tree
tree.plot_tree(tree_r)