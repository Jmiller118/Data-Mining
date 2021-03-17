#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

import csv

Pkl_Filename = "SVM_Assignment02.pkl"  

# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file, encoding='bytes')

test = pd.read_csv("test.csv", sep=',', header=None)

features = pd.DataFrame()

maxV = test.max(axis=1) 
minV = test.min(axis=1)
featureVector_Vals = maxV - minV
features['F1'] = featureVector_Vals

featureVector_Max = test.idxmax(axis=1)
features['F2'] = featureVector_Max

timeVals = []
for x,y in featureVector_Max.iteritems():
    mins = 5 + (5 * y)
    timeVals.append(mins)
timeVals = pd.DataFrame(timeVals)

vel = []
for index, row in maxV.iteritems():
    for i, r in timeVals.iteritems():
        if index == i:
            velocity = (row/r)
            vel.append(velocity)
        
featureVector_Vel = pd.DataFrame(vel)
featureVector_Vel = featureVector_Vel.T
features['F3'] = featureVector_Vel



Ypredict = Pickled_LR_Model.predict(features)  
Ypredict = pd.DataFrame(Ypredict)



# In[16]:



print(Ypredict)


# In[ ]:





# In[ ]:




