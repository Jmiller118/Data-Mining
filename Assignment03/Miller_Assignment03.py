#!/usr/bin/env python
# coding: utf-8

# In[214]:


import datetime as datetime
import pandas as pd
import numpy as np
import csv
from math import log

from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import DBSCAN 
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
#from scipy.special import entr
from sklearn.decomposition import PCA 
from scipy.stats import entropy


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


# In[215]:


insulin_data = pd.read_csv("InsulinData.csv")
insulin_data2 = pd.read_csv("InsulinAndMealIntake670GPatient3.csv")

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
#print(insulin.head(50))

insulin = insulin.set_index("Time")
insulin2 = insulin2.set_index("Time")

insulin.columns = ['data']
insulin2.columns = ['data']

insulin = insulin.iloc[::-1]
insulin2 = insulin2.iloc[::-1]



# In[216]:


meal_Time = pd.DataFrame(columns = ['startTime', 'endTime'])
noMeal_Time = pd.DataFrame(columns = ['startTime', 'endTime'])

yValue = pd.DataFrame(columns =['startTime', 'endTime', 'YValue'])


# In[217]:


if pd.isnull(insulin.iloc[0,0]):
    startTime = insulin.index[0]
    endTime = startTime + pd.offsets.Minute(120)
    curr = pd.DataFrame()
    i = 0
        
    while ((endTime < insulin.index[len(insulin)-1]) & (i < 41434)):
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
                        y_Val = curr.iloc[j,0]

                curr = insulin[(insulin.index > startTime)]
                curr = curr[(curr.index < endTime)]
                
                if ((pd.notnull(curr['data'])) & (curr['data'] != 0)).any():
                    mealNext = True
            #meal_Time = meal_Time.append({'startTime': startMThirty, 'endTime': endTime}, ignore_index=True)
            yValue = yValue.append({'startTime': startMThirty, 'endTime': endTime, 'YValue': y_Val}, ignore_index=True)
            i = i + 1
            startTime = endTime
            endTime = startTime + pd.offsets.Minute(120)


# In[218]:


#print(yValue)
meal_Time2 = pd.DataFrame(columns = ['startTime', 'endTime'])
noMeal_Time2 = pd.DataFrame(columns = ['startTime', 'endTime'])
yValue2 = pd.DataFrame(columns =['startTime', 'endTime', 'YValue'])


# In[219]:


if pd.isnull(insulin2.iloc[0,0]):
    startTime = insulin2.index[0]
    endTime = startTime + pd.offsets.Minute(120)
    curr = pd.DataFrame()
    i = 0
    
    while ((endTime < insulin2.index[len(insulin2)-1]) & (i < 41434)):
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
                        y_Val2 = curr.iloc[j,0]
                curr = insulin2[(insulin2.index > startTime)]
                curr = curr[(curr.index < endTime)]
                
                if ((pd.notnull(curr['data'])) & (curr['data'] != 0)).any():
                    mealNext = True
            #meal_Time2 = meal_Time2.append({'startTime': startMThirty, 'endTime': endTime}, ignore_index=True)
            yValue2 = yValue2.append({'startTime': startMThirty, 'endTime': endTime, 'YValue': y_Val2}, ignore_index=True)

            i = i + 1
            startTime = endTime
            endTime = startTime + pd.offsets.Minute(120)


# In[220]:



meal_Sensor = pd.DataFrame()
noMeal_Sensor = pd.DataFrame()


# In[221]:


y_val = []
for i in range(0, yValue.index[len(yValue)-1]):
    startTime = yValue.iloc[i,0]
    endTime = yValue.iloc[i,1]
    yval = yValue.iloc[i,2]
    y_val.append(yval)
    
    curr = pd.DataFrame()
    curr = cgm[(cgm.index < endTime)]
    curr = curr[(curr.index >= startTime)]
    curr = curr.reset_index(drop=True)
    curr = curr.T
    meal_Sensor = pd.concat([meal_Sensor, curr], ignore_index=True)
      
yval = pd.DataFrame()
yval['31'] = y_val


# In[222]:


meal_Sensor = pd.concat([meal_Sensor, yval], axis=1)


# In[223]:


y_val2 = []

for i in range(0, yValue2.index[len(meal_Time2)-1]):
    startTime = yValue2.iloc[i,0]
    endTime = yValue2.iloc[i,1]
    yval2 = yValue2.iloc[i,2]
    y_val2.append(yval2)
    
    curr = pd.DataFrame()
    curr = cgm2[(cgm2.index < endTime)]
    curr = curr[(curr.index >= startTime)]
    curr = curr.reset_index(drop=True)
    curr = curr.T
    meal_Sensor = pd.concat([meal_Sensor, curr], ignore_index=True)
    
#print(meal_Sensor)
       
yval2 = pd.DataFrame()
yval2['31'] = y_val2    


# In[224]:



meal_Sensor.append(yval2, ignore_index=True, sort=False)
meal_Sensor = meal_Sensor.dropna()


# In[225]:


meal = meal_Sensor
meal.to_csv("meal_Data_COPY.csv")


# In[226]:


data_ = pd.read_csv('meal_Data_COPY.csv', sep=',' ,header=None)

Y = data_.iloc[:, [31]]
data = data_.drop(data_.index[31], axis=1)
Y = Y.iloc[1:]


# In[227]:


mealVector = pd.DataFrame()


# In[228]:


#get the max value of the cgm data
maxVal = data.max(axis=1)
#get the min value of the cgm data
minVal = data.min(axis=1)

#create a feature vec of CGMmax - CGMmin
featureVector_CGM_Meal = maxVal - minVal

mealVector['F1'] = featureVector_CGM_Meal


# In[229]:


maxValLoc = data.idxmax(axis=1)
#feature vec of time for cgm max data
featureVector_CGM_MaxTime_Meal = maxValLoc
mealVector['F2'] = featureVector_CGM_MaxTime_Meal


# In[230]:


#tells the time in intervals of 5
timeVals = []
for x,y in featureVector_CGM_MaxTime_Meal.iteritems():
    mins = 5 + (5 * y)
    timeVals.append(mins)
    
timeVals = pd.DataFrame(timeVals)
 


# In[231]:


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


# In[232]:


mealVector['F4'] = Y
final = pd.DataFrame()


# In[233]:


max1 = mealVector['F4'].max()
#print(max1)

min1 = mealVector['F4'].min()
#print(min1)


# In[234]:



bins=[0, 20, 40, 60, 80, 100, 120, 140] # 200, 220]

mealVector['bin'] = pd.cut(mealVector['F4'], bins=bins, labels=False)
mealVectorVals = mealVector.values


# In[235]:


mealVector = mealVector.fillna(0)


# In[254]:


X = mealVector.iloc[:, [0,1,2]]

N = X.shape[0]
s = X

fft = np.fft.fft(s)
freq = np.fft.fftfreq(len(s))

#print(fft, freq)

freq = freq.reshape(-1,1)


plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.plot(freq,fft)
plt.show()


# In[388]:


bins = mealVector.iloc[:, [1, 2, 3]]
bins = bins.to_numpy()

bins = bins.tolist()

km = KMeans(
    n_clusters=6, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)

y_km = km.fit_predict(freq)
#print(y_km)

y_km2 = km.fit_predict(bins)
#print(y_km2)


# In[389]:


plt.scatter(
    mealVectorVals[y_km == 0, 0], mealVectorVals[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='o', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    mealVectorVals[y_km == 1, 0], mealVectorVals[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    mealVectorVals[y_km == 2, 0], mealVectorVals[y_km == 2, 1],
    s=50, c='lightblue',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    mealVectorVals[y_km == 3, 0], mealVectorVals[y_km == 3, 1],
    s=50, c='pink',
    marker='o', edgecolor='black',
    label='cluster 3'
)

plt.scatter(
    mealVectorVals[y_km == 4, 0], mealVectorVals[y_km == 4, 1],
    s=50, c='green',
    marker='o', edgecolor='black',
    label='cluster 4'
)

plt.scatter(
    mealVectorVals[y_km == 5, 0], mealVectorVals[y_km == 5, 1],
    s=50, c='blue',
    marker='o', edgecolor='black',
    label='cluster 5'
)






# In[390]:


plt.scatter(
    mealVectorVals[y_km2 == 0, 0], mealVectorVals[y_km2 == 0, 1],
    s=50, c='lightgreen',
    marker='o', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    mealVectorVals[y_km2 == 1, 0], mealVectorVals[y_km2 == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    mealVectorVals[y_km2 == 2, 0], mealVectorVals[y_km2 == 2, 1],
    s=50, c='lightblue',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    mealVectorVals[y_km2 == 3, 0], mealVectorVals[y_km2 == 3, 1],
    s=50, c='pink',
    marker='o', edgecolor='black',
    label='cluster 3'
)

plt.scatter(
    mealVectorVals[y_km2 == 4, 0], mealVectorVals[y_km2 == 4, 1],
    s=50, c='green',
    marker='o', edgecolor='black',
    label='cluster 4'
)

plt.scatter(
    mealVectorVals[y_km2 == 5, 0], mealVectorVals[y_km2 == 5, 1],
    s=50, c='blue',
    marker='o', edgecolor='black',
    label='cluster 5'
)


# In[391]:


distortions = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(freq)
    distortions.append(km.inertia_)
    

# plot
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()


# In[405]:


#DBSCAN
scan = mealVector.iloc[:, [1, 2, 3]]
scan = scan.to_numpy()
scan = scan.tolist()
#print(scan.to_numpy().tolist())

"""
N = scan.shape[0]
s = X

fft2 = np.fft.fft(s)
freq2 = np.fft.fftfreq(len(s))


freq2 = freq2.reshape(-1,1)

print(freq2)
"""
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(scan) 
  
# Normalizing the data so that  
# the data approximately follows a Gaussian distribution 
X_normalized = normalize(X_scaled) 
  
# Converting the numpy array into a pandas DataFrame 
X_normalized = pd.DataFrame(X_normalized) 

#X_normalized.values


# In[406]:


pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(X_normalized) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 

db_default = DBSCAN(eps = 0.0375, min_samples = 3).fit(X_principal) 
y_db = db_default.fit_predict(X_principal)
#print(y_db)
labels = db_default.labels_ 
#print(labels)


# In[407]:


colors = {} 
colors[0] = 'red'
colors[1] = 'green'
colors[2] = 'blue'
colors[3] = 'cyan'
colors[4] = 'purple'
colors[5] = 'tomato'
colors[6] = 'olive'
colors[7] = 'blueviolet'
colors[8] = 'lightgreen'
colors[9] = 'hotpink'
colors[10] = 'brown'
colors[11] = 'fuchsia'
colors[12] = 'gold'
colors[13] = 'orange'
colors[14] = 'teal'
colors[15] = 'deeppink'
colors[16] = 'silver'
colors[17] = 'thistle'
colors[18] = 'chocolate'
colors[19] = 'darkorange'
colors[20] = 'slateblue'
colors[21] = 'dodgerblue'
colors[22] = 'yellowgreen'
colors[23] = 'sandybrown'
colors[24] = 'azure'
colors[25] = 'springgreen'
colors[26] = 'indigo'
colors[27] = 'plum'
colors[28] = 'firebrick'
colors[29] = 'skyblue'
colors[30] = 'violet'
colors[31] = 'aquamarine'
colors[-1] = 'coral'
  
# Building the colour vector for each data point 
cvec = [colors[label] for label in labels] 

plt.figure(figsize =(9, 9)) 
plt.scatter(X_principal['P1'], X_principal['P2'] , c = cvec) 
 
plt.show() 
   


# In[408]:


plt.figure(figsize=(10,5))
nn = NearestNeighbors(n_neighbors=5).fit(X_normalized)
distances, idx = nn.kneighbors(X_normalized)
distances = np.sort(distances, axis=0)
distances = distances[:,1]

plt.plot(distances)
plt.show()


#Computing "the Silhouette Score"
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X_principal, labels))



# In[409]:


km.inertia_
final['KMeans SSE'] = km.inertia_
#print(final)

silhoutte = metrics.silhouette_score(X_principal, labels)
final['DBScan Silhoette Score'] = silhoutte

#***********************
ent = entropy(mealVector['bin'], base=2)
final['Kmeans Entropy'] = ent 


ent2 = entropy(mealVector['bin'], base=2)
final['DBScan Entropy'] = ent2



#***********************
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

purity = purity_score(mealVector['bin'], km.labels_)
#purity = purity_score(y_km, km.labels_)
pur= []
pur.append(purity)
final['Kmeans Purity'] = pur

purity2 = purity_score(mealVector['bin'], db_default.labels_)
pur2 = []
pur2.append(purity2)
final['DBScan Purity'] = pur2


# In[410]:


rows = [final]
file = pd.concat(rows, axis=0)
file.to_csv("Miller_Assignment03_Results.csv")


# In[ ]:




