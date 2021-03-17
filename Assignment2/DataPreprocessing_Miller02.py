#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime as datetime
import pandas as pd
import numpy as np
import pickle
import math
import csv
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.svm import SVC, SVR
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from scipy.fft import fft

cgm_data = pd.read_csv('CGMData.csv')
cgm_data2 = pd.read_csv('CGMData670GPatient2.csv')

cgm_data.columns = [c.replace(' ', '_') for c in cgm_data.columns]
cgm_data2.columns = [c.replace(' ', '_') for c in cgm_data2.columns]

cgm = cgm_data[["Date", "Time", "Sensor_Glucose_(mg/dL)"]]
cgm2 = cgm_data2[["Date", "Time", "Sensor_Glucose_(mg/dL)"]]

cgm["Time"] = cgm["Date"] + ' ' + cgm["Time"]
cgm2["Time"] = cgm2["Date"] + ' ' + cgm2["Time"]

cgm["Time"] = cgm["Time"].astype("datetime64[ns]")
cgm = cgm[['Time', 'Sensor_Glucose_(mg/dL)']]

cgm2["Time"] = cgm2["Time"].astype("datetime64[ns]")
cgm2 = cgm2[["Time", "Sensor_Glucose_(mg/dL)"]]

cgm = cgm.set_index("Time")
cgm2 = cgm2.set_index("Time")

cgm = cgm.iloc[::-1]
cgm2 = cgm2.iloc[::-1]


# In[2]:


insulin_data = pd.read_csv("InsulinData.csv")
insulin_data2 = pd.read_csv("InsulinAndMealIntake670GPatient2.csv")

insulin_data.columns = [c.replace(' ', '_') for c in insulin_data.columns]
insulin_data2.columns = [c.replace(' ', '_') for c in insulin_data2.columns]

insulin = insulin_data[["Date", "Time", "BWZ_Carb_Input_(grams)"]]
insulin2 = insulin_data2[["Date", "Time", "BWZ_Carb_Input_(grams)"]]

insulin["Time"] = insulin["Date"] + ' ' + insulin["Time"]
insulin2["Time"] = insulin2["Date"] + ' ' + insulin2["Time"]

insulin["Time"] = insulin["Time"].astype("datetime64[ns]")
insulin2["Time"] = insulin2["Time"].astype("datetime64[ns]")

insulin = insulin[["Time", "BWZ_Carb_Input_(grams)"]]
insulin2 = insulin2[["Time", "BWZ_Carb_Input_(grams)"]]

insulin = insulin.set_index("Time")
insulin2 = insulin2.set_index("Time")

insulin.columns = ['data']
insulin2.columns = ['data']

insulin = insulin.iloc[::-1]
insulin2 = insulin2.iloc[::-1]


# In[3]:


insulin = insulin[(insulin.index > cgm.index[0])]
insulin2 = insulin2[(insulin2.index > cgm2.index[0])]


# In[4]:


meal_Time = pd.DataFrame(columns = ['startTime', 'endTime'])
noMeal_Time = pd.DataFrame(columns = ['startTime', 'endTime'])


# In[5]:


if pd.isnull(insulin.iloc[0,0]):
    startTime = insulin.index[0]
    endTime = startTime + pd.offsets.Minute(120)
    curr = pd.DataFrame()
    i = 0
    
    while ((endTime < insulin.index[len(insulin)-1]) & (i < 150)):
        curr = insulin[(insulin.index >= startTime)]
        curr = curr[(curr.index < endTime)]
        hasMeal = False
        
        if ((pd.notnull(curr['data'])) & (curr['data'] != 0)).any():
            hasMeal = True
        
        if (hasMeal == False):
            noMeal_Time = noMeal_Time.append({'startTime': startTime, 'endTime': endTime}, ignore_index=True)
            startTime = endTime
            endTime = startTime + pd.offsets.Minute(120)
            
        elif (hasMeal == True):
            mealNext = True
            while ((mealNext == True) & (endTime < insulin.index[len(insulin)-1])):
                mealNext = False
                for j in range(len(curr)):
                    if ((pd.notnull(curr.iloc[j,0])) and (curr.iloc[j,0] != 0)):
                        startTime = curr.index[j]
                        startMThirty = startTime - pd.offsets.Minute(30)
                        endTime = startTime + pd.offsets.Minute(120)
                curr = insulin[(insulin.index > startTime)]
                curr = curr[(curr.index < endTime)]
                
                if ((pd.notnull(curr['data'])) & (curr['data'] != 0)).any():
                    mealNext = True
            meal_Time = meal_Time.append({'startTime': startMThirty, 'endTime': endTime}, ignore_index=True)
            i = i + 1
            startTime = endTime
            endTime = startTime + pd.offsets.Minute(120)
            
print(meal_Time)            
print(noMeal_Time)


# In[6]:


meal_Time2 = pd.DataFrame(columns = ['startTime', 'endTime'])
noMeal_Time2 = pd.DataFrame(columns = ['startTime', 'endTime'])


# In[7]:


if pd.isnull(insulin2.iloc[0,0]):
    startTime = insulin2.index[0]
    endTime = startTime + pd.offsets.Minute(120)
    curr = pd.DataFrame()
    i = 0
    
    while ((endTime < insulin2.index[len(insulin2)-1]) & (i < 150)):
        curr = insulin2[(insulin2.index >= startTime)]
        curr = curr[(curr.index < endTime)]
        hasMeal = False
        
        if ((pd.notnull(curr['data'])) & (curr['data'] != 0)).any():
            hasMeal = True
        
        if (hasMeal == False):
            noMeal_Time2 = noMeal_Time2.append({'startTime': startTime, 'endTime': endTime}, ignore_index=True)
            startTime = endTime
            endTime = startTime + pd.offsets.Minute(120)
            
        elif (hasMeal == True):
            mealNext = True
            while ((mealNext == True) & (endTime < insulin2.index[len(insulin2)-1])):
                mealNext = False
                for j in range(len(curr)):
                    if ((pd.notnull(curr.iloc[j,0])) and (curr.iloc[j,0] != 0)):
                        startTime = curr.index[j]
                        startMThirty = startTime - pd.offsets.Minute(30)
                        endTime = startTime + pd.offsets.Minute(120)
                curr = insulin2[(insulin2.index > startTime)]
                curr = curr[(curr.index < endTime)]
                
                if ((pd.notnull(curr['data'])) & (curr['data'] != 0)).any():
                    mealNext = True
            meal_Time2 = meal_Time2.append({'startTime': startMThirty, 'endTime': endTime}, ignore_index=True)
            i = i + 1
            startTime = endTime
            endTime = startTime + pd.offsets.Minute(120)
            
#print(meal_Time2)            
#print(noMeal_Time2)
            


# In[8]:


#meal_Time.to_csv("meal_Time.csv")
#noMeal_Time.to_csv('nomeal_Time.csv')


# In[9]:


meal_Sensor = pd.DataFrame()
noMeal_Sensor = pd.DataFrame()


# In[10]:


for i in range(0, meal_Time.index[len(meal_Time)-1]):
    startTime = meal_Time.iloc[i,0]
    endTime = meal_Time.iloc[i,1]
    
    curr = pd.DataFrame()
    curr = cgm[(cgm.index < endTime)]
    curr = curr[(curr.index >= startTime)]
    curr = curr.reset_index(drop=True)
    curr = curr.T
    meal_Sensor = pd.concat([meal_Sensor, curr], ignore_index=True)
    
#print(meal_Sensor)


# In[11]:


for i in range(0, meal_Time2.index[len(meal_Time2)-1]):
    startTime = meal_Time2.iloc[i,0]
    endTime = meal_Time2.iloc[i,1]
    
    curr = pd.DataFrame()
    curr = cgm2[(cgm2.index < endTime)]
    curr = curr[(curr.index >= startTime)]
    curr = curr.reset_index(drop=True)
    curr = curr.T
    meal_Sensor = pd.concat([meal_Sensor, curr], ignore_index=True)
    
#print(meal_Sensor)


# In[12]:


for i in range(0, noMeal_Time.index[len(noMeal_Time)-1]):
    startTime = noMeal_Time.iloc[i,0]
    endTime = noMeal_Time.iloc[i,1]
    
    curr = pd.DataFrame()
    curr = cgm[(cgm.index < endTime)]
    curr = curr[(curr.index >= startTime)]
    curr = curr.reset_index(drop=True)
    curr = curr.T
    noMeal_Sensor = pd.concat([noMeal_Sensor, curr], ignore_index=True)
    
#print(noMeal_Sensor)


# In[13]:


for i in range(0, noMeal_Time2.index[len(noMeal_Time2)-1]):
    startTime = noMeal_Time2.iloc[i,0]
    endTime = noMeal_Time2.iloc[i,1]
    
    curr = pd.DataFrame()
    curr = cgm2[(cgm2.index < endTime)]
    curr = curr[(curr.index >= startTime)]
    curr = curr.reset_index(drop=True)
    curr = curr.T
    noMeal_Sensor = pd.concat([noMeal_Sensor, curr], ignore_index=True)
    
#print(noMeal_Sensor)



# In[14]:


meal = meal_Sensor.dropna()
noMeal = noMeal_Sensor.fillna(0)

#print(meal)
#print(noMeal)


meal.to_csv("meal_Data.csv")
noMeal.to_csv("noMeal_Data.csv")


# In[15]:


data = pd.read_csv('meal_Data.csv',sep=',',header=None)
data2 = pd.read_csv('noMeal_Data.csv',sep=',',header=None)

#print(data, data2)


# In[16]:


mealVector = pd.DataFrame()
noMealVector = pd.DataFrame()


# In[17]:


#get the max value of the cgm data
maxVal = data.max(axis=1)
#get the min value of the cgm data
minVal = data.min(axis=1)
#create a feature vec of CGMmax - CGMmin
featureVector_CGM_Meal = maxVal - minVal
mealVector['F1'] = featureVector_CGM_Meal

#****************************
#get the max value of the cgm data
maxVal2 = data2.max(axis=1)
#get the min value of the cgm data
minVal2 = data2.min(axis=1)
#create a feature vec of CGMmax - CGMmin
featureVector_CGM_NoMeal = maxVal2 - minVal2
noMealVector['F1'] = featureVector_CGM_NoMeal


# In[18]:


maxValLoc = data.idxmax(axis=1)
#feature vec of time for cgm max data
featureVector_CGM_MaxTime_Meal = maxValLoc
mealVector['F2'] = featureVector_CGM_MaxTime_Meal

#****************************
maxValLoc2 = data2.idxmax(axis=1)
#feature vec of time for cgm max data
featureVector_CGM_MaxTime_NoMeal = maxValLoc2
noMealVector['F2'] = featureVector_CGM_MaxTime_NoMeal


# In[19]:


#tells the time in intervals of 5
timeVals = []
for x,y in featureVector_CGM_MaxTime_Meal.iteritems():
    mins = 5 + (5 * y)
    timeVals.append(mins)

timeVals = pd.DataFrame(timeVals)
  
#****************************
#tells the time in intervals of 5
timeVals2 = []
for x,y in featureVector_CGM_MaxTime_NoMeal.iteritems():
    mins = 5 + (5 * y)
    timeVals2.append(mins)
    
timeVals2 = pd.DataFrame(timeVals2)


# In[20]:


#calc velocity for meal
vel = []
for index, row in maxVal.iteritems():
    for i, r in timeVals.iteritems():
        if index == i:
            velocity = (row/r)
            vel.append(velocity)
        
featureVector_Meal_Vel = pd.DataFrame(vel)
featureVector_Meal_Vel = featureVector_Meal_Vel.T
mealVector['F3'] = featureVector_Meal_Vel

#***************************
#calc velocity for no meal
vel2 = []
for index, row in maxVal2.iteritems():
    for i, r in timeVals2.iteritems():
        if index == i:
            velocity2 = (row/r)
            vel2.append(velocity2)
        
featureVector_NoMeal_Vel = pd.DataFrame(vel2)
featureVector_NoMeal_Vel = featureVector_NoMeal_Vel.T
noMealVector['F3'] = featureVector_NoMeal_Vel


# In[33]:


#add ground truth
labelVectorOnes = np.ones(len(mealVector))
labelVectorOnes = pd.DataFrame(labelVectorOnes)
mealVector['Truth'] = labelVectorOnes

labelVectorZeros = np.zeros(len(noMealVector))
labelVectorZeros = pd.DataFrame(labelVectorZeros)
noMealVector['Truth'] = labelVectorZeros

noMealUU = noMealVector.head(243)
#print(noMealUU)

#print(len(mealVector),  len(noMealUU))

"""
#append them and shuffle
frames = pd.DataFrame()
frames = mealVector.append(noMealVector)
frames = shuffle(frames)

#seperate features from labels
X = frames.iloc[:, [0, 1,2]]
y = frames.iloc[:, [3]]

"""
frames = pd.DataFrame()
frames = mealVector.append(noMealUU)
frames = shuffle(frames)
X = frames.iloc[:, [0, 1,2]]
y = frames.iloc[:, [3]]

print(X.shape)


# In[31]:



N = X.shape[0]
s = X

fft = np.fft.fft(s)
freq = np.fft.fftfreq(len(s))

print(len(freq))

plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.plot(freq,fft)
plt.show()
"""
"""


# In[23]:


#grid search
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.001, 0.0001], 
              'kernel': ['linear', 'rbf']}

#freq = freq.reshape(-1,1)

#print(type(freq))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#print(type(X_test))
#print(type(y))


# In[24]:



scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), param_grid, n_jobs = -1, scoring='%s_macro' % score, verbose = 3
    )
    clf.fit(X_train, y_train.values.ravel())
    #clf.fit(X_train, y_train)


    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(metrics.classification_report(y_true, y_pred))
    print(metrics.confusion_matrix(y_true, y_pred))
    print()


# In[25]:


SVC_scores = cross_val_score(clf, X, y.values.ravel(), cv = 30)
print(SVC_scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (SVC_scores.mean(), SVC_scores.std() * 2))


# In[29]:


Pkl_Filename = "Best_SVM_Final.pkl"  

print(clf.best_params_)

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf, file)
    
file.close()


# In[ ]:





# In[ ]:




