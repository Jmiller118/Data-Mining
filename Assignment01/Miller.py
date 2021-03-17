#!/usr/bin/env python
# coding: utf-8

# In[301]:


import pandas as pd
import numpy as np
import csv
from datetime import datetime

#all data, read in cgm data
cgm_data = pd.read_csv('CGMData.csv')
cgm = cgm_data[["Date", "Time", "SensorGlucose"]]

#make date & time, datetime
cgm["Time"] = cgm["Date"] + ' ' + cgm["Time"]
cgm["Time"] = cgm["Time"].astype("datetime64[ns]")

#make new based on the combo
cgm = cgm[["Time", "SensorGlucose"]]
cgm = cgm.set_index("Time")

#all insulin data
insulin_data = pd.read_csv("InsulinData.csv", dtype={"Alarm":"string"})
insulin = insulin_data[["Date", "Time", "Alarm"]]

#make date & time, datetime
insulin["Time"] = insulin["Date"] + ' ' + insulin["Time"]
insulin["Time"] = insulin["Time"].astype("datetime64[ns]")

#set the old to the new
insulin = insulin[["Time", "Alarm"]]
insulin = insulin.set_index("Time")

#find where it goes from manual to auto in insulin
insulinChange = insulin[insulin['Alarm'].str.contains("AUTO MODE ACTIVE PLGM OFF")]

#split the cgm data
cgm_manual = cgm[(cgm.index < insulinChange.index.min())]
cgm_auto = cgm[(cgm.index > insulinChange.index.min())]

#Using linear interpolation for filling the missing values
cgm_manual['SensorGlucose'] = cgm_manual['SensorGlucose'].interpolate(method='linear')
cgm_auto['SensorGlucose'] = cgm_auto['SensorGlucose'].interpolate(method='linear')

threshold1 = [0, 53.99]
threshold2 = [0, 69.99]
threshold3 = [70, 150]
threshold4 = [70, 180]
threshold5 = [180, 100000]
threshold6 = [250, 100000]


# In[302]:


#####################################################################

#daily counts
man1 = cgm_manual.resample("D")["SensorGlucose"].value_counts(bins=threshold1, sort=False).to_frame(name="Whole Day percentage in Hyperglycemia Level 2 (CGM < 54)")
man2 = cgm_manual.resample("D")['SensorGlucose'].value_counts(bins=threshold2, sort=False).to_frame(name="Whole Day percentage in Hyperglycemia Level 1 (CGM < 70)")
man3 = cgm_manual.resample("D")['SensorGlucose'].value_counts(bins=threshold3, sort=False).to_frame(name="Whole Day percentage in secondary range 70 <= CGM <= 150")
man4 = cgm_manual.resample("D")['SensorGlucose'].value_counts(bins=threshold4, sort=False).to_frame(name="Whole Day percentage in first range 70 <= CGM <= 180")
man5 = cgm_manual.resample("D")["SensorGlucose"].value_counts(bins=threshold5, sort=False).to_frame(name="Whole Day percentage in Hyperglycemia CGM > 180")
man6 = cgm_manual.resample("D")["SensorGlucose"].value_counts(bins=threshold6, sort=False).to_frame(name="Whole Day percentage in Critical Hyperglycemia CGM > 250")


# In[303]:


#####################################################################
#sum the daily counts
manualSum = pd.merge(man1, man2, on="Time", how="outer", sort=True)
manualSum = pd.merge(manualSum, man3, on='Time', how='outer', sort=True)
manualSum = pd.merge(manualSum, man4, on="Time", how="outer", sort=True)
manualSum = pd.merge(manualSum, man5, on="Time", how="outer", sort=True)
manualSum = pd.merge(manualSum, man6, on="Time", how='outer', sort=True)


# In[304]:


manualSum = manualSum.fillna(0)

totalSumDaily = manualSum


# In[305]:


#calc percentages
add = totalSumDaily['Whole Day percentage in Hyperglycemia Level 2 (CGM < 54)']
totalSumDaily['Whole Day percentage in Hyperglycemia Level 2 (CGM < 54)'] = add

add = totalSumDaily['Whole Day percentage in Hyperglycemia Level 1 (CGM < 70)'] 
totalSumDaily['Whole Day percentage in Hyperglycemia Level 1 (CGM < 70)'] = add

add = totalSumDaily['Whole Day percentage in secondary range 70 <= CGM <= 150']
totalSumDaily['Whole Day percentage in secondary range 70 <= CGM <= 150)'] = add

add = totalSumDaily['Whole Day percentage in first range 70 <= CGM <= 180']
totalSumDaily['Whole Day percentage in first range 70 <= CGM <= 180'] = add

add = totalSumDaily['Whole Day percentage in Hyperglycemia CGM > 180']
totalSumDaily['Whole Day percentage in Hyperglycemia CGM > 180'] = add

add = totalSumDaily["Whole Day percentage in Critical Hyperglycemia CGM > 250"]
totalSumDaily['Whole Day percentage in Critical Hyperglycemia CGM > 250'] = add


# In[306]:


#average over 24 hours
manual_daily_avg = totalSumDaily.mean() / 288 * 100
df = pd.DataFrame(manual_daily_avg)


# In[307]:


results = df.T.rename(index={0:"Manual Mode"})


# In[308]:


#so we can merge into hw later
results0 = results[['Whole Day percentage in Hyperglycemia CGM > 180', 
    "Whole Day percentage in Critical Hyperglycemia CGM > 250",
    'Whole Day percentage in first range 70 <= CGM <= 180', 
    'Whole Day percentage in secondary range 70 <= CGM <= 150', 
    'Whole Day percentage in Hyperglycemia Level 1 (CGM < 70)', 
    'Whole Day percentage in Hyperglycemia Level 2 (CGM < 54)']]


# In[309]:


#################################################################################
#6 to midnight
day = cgm_manual.index.hour
mask = (day >= 6) & (day < 24)

#find out how many
man1_day = cgm_manual[mask].groupby(cgm_manual[mask].index.floor('D'))["SensorGlucose"].value_counts(bins=threshold1, sort=False).to_frame(name="Daytime percentage in Hyperglycemia Level 2 (CGM < 54)")
man2_day = cgm_manual[mask].groupby(cgm_manual[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold2, sort=False).to_frame(name="Daytime percentage in Hyperglycemia Level 1 (CGM < 70)")
man3_day = cgm_manual[mask].groupby(cgm_manual[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold3, sort=False).to_frame(name="Daytime percentage in Secondary Range 70 <= CGM <= 150")
man4_day = cgm_manual[mask].groupby(cgm_manual[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold4, sort=False).to_frame(name="Daytime percentage in First Range 70 <= CGM <= 180")
man5_day = cgm_manual[mask].groupby(cgm_manual[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold5, sort=False).to_frame(name="Daytime percentage in Hyperglycemia CGM > 180")
man6_day = cgm_manual[mask].groupby(cgm_manual[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold6, sort=False).to_frame(name="Daytime percentage in Critical Hyperglycemia CGM > 250")


# In[310]:


#sum them up
manualSum = pd.merge(man1_day, man2_day, on="Time", how="outer", sort=True)
manualSum = pd.merge(manualSum, man3_day, on='Time', how='outer', sort=True)
manualSum = pd.merge(manualSum, man4_day, on="Time", how="outer", sort=True)
manualSum = pd.merge(manualSum, man5_day, on="Time", how="outer", sort=True)
manualSum = pd.merge(manualSum, man6_day, on="Time", how='outer', sort=True)

manualSum = manualSum.fillna(0)

#give them a placeholder
manualSum = manualSum[['Daytime percentage in Hyperglycemia Level 2 (CGM < 54)', 
    'Daytime percentage in Hyperglycemia Level 1 (CGM < 70)', 
    'Daytime percentage in Secondary Range 70 <= CGM <= 150', 
    'Daytime percentage in First Range 70 <= CGM <= 180', 
    'Daytime percentage in Hyperglycemia CGM > 180', 
    'Daytime percentage in Critical Hyperglycemia CGM > 250']]

sumManualDaytime = manualSum


# In[311]:


#get the percentage
add = sumManualDaytime['Daytime percentage in Hyperglycemia Level 2 (CGM < 54)']
totalSumDaily['Daytime percentage in Hyperglycemia Level 2 (CGM < 54)'] = add

add = sumManualDaytime['Daytime percentage in Hyperglycemia Level 1 (CGM < 70)']
totalSumDaily['Daytime percentage in HyperL1 Count'] = add

add = sumManualDaytime['Daytime percentage in Secondary Range 70 <= CGM <= 150']
totalSumDaily['Daytime percentage in Secondary Range 70 <= CGM <= 150'] = add

add = sumManualDaytime['Daytime percentage in First Range 70 <= CGM <= 180']
totalSumDaily['Daytime percentage in First Range 70 <= CGM <= 180'] = add

add = sumManualDaytime['Daytime percentage in Hyperglycemia CGM > 180'] 
totalSumDaily['Daytime percentage in Hyperglycemia > 180'] = add

add = sumManualDaytime['Daytime percentage in Critical Hyperglycemia CGM > 250']
totalSumDaily['Daytime percentage in Critical Hyperglycemia CGM > 250'] = add

manual_daily_avg = sumManualDaytime.mean() / 288 * 100
df = pd.DataFrame(manual_daily_avg)
results = df.T.rename(index={0: "Manual Mode"})

#placeholder
results1 = results[['Daytime percentage in Hyperglycemia CGM > 180', 
    "Daytime percentage in Critical Hyperglycemia CGM > 250",
    'Daytime percentage in First Range 70 <= CGM <= 180', 
    'Daytime percentage in Secondary Range 70 <= CGM <= 150', 
    'Daytime percentage in Hyperglycemia Level 1 (CGM < 70)', 
    'Daytime percentage in Hyperglycemia Level 2 (CGM < 54)']]


# In[312]:


#find the nightime hours!
night = cgm_manual.index.hour
mask = (night >= 0) & (night < 6) 

man1_night = cgm_manual[mask].groupby(cgm_manual[mask].index.floor('D'))["SensorGlucose"].value_counts(bins=threshold1, sort=False).to_frame(name="Nighttime percentage in Hyperglycemia Level 2 (CGM < 54)")
man2_night = cgm_manual[mask].groupby(cgm_manual[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold2, sort=False).to_frame(name="Nighttime percentage in Hyperglycemia Level 1 (CGM < 70)")
man3_night = cgm_manual[mask].groupby(cgm_manual[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold3, sort=False).to_frame(name="Nighttime percentage in Secondary Range 70 <= CGM <= 150")
man4_night = cgm_manual[mask].groupby(cgm_manual[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold4, sort=False).to_frame(name="Nighttime percentage in First Range 70 <= CGM <= 180")
man5_night = cgm_manual[mask].groupby(cgm_manual[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold5, sort=False).to_frame(name="Nighttime percentage in Hyperglycemia CGM > 180")
man6_night = cgm_manual[mask].groupby(cgm_manual[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold6, sort=False).to_frame(name="Nighttime percentage in Critical Hyperglycemia CGM > 250")


# In[313]:


manualSum = pd.merge(man1_night, man2_night, on="Time", how="outer", sort=True)
manualSum = pd.merge(manualSum, man3_night, on='Time', how='outer', sort=True)
manualSum = pd.merge(manualSum, man4_night, on="Time", how="outer", sort=True)
manualSum = pd.merge(manualSum, man5_night, on="Time", how="outer", sort=True)
manualSum = pd.merge(manualSum, man6_night, on="Time", how='outer', sort=True)

manualSum = manualSum.fillna(0)

manualSum = manualSum[['Nighttime percentage in Hyperglycemia Level 2 (CGM < 54)', 
    'Nighttime percentage in Hyperglycemia Level 1 (CGM < 70)', 
    'Nighttime percentage in Secondary Range 70 <= CGM <= 150', 
    'Nighttime percentage in First Range 70 <= CGM <= 180', 
    'Nighttime percentage in Hyperglycemia CGM > 180', 
    'Nighttime percentage in Critical Hyperglycemia CGM > 250']]


sumManualNighttime = manualSum


# In[314]:


#perctanges
add = sumManualNighttime['Nighttime percentage in Hyperglycemia Level 2 (CGM < 54)']
sumManualNighttime['Nighttime percentage in Hyperglycemia Level 2 (CGM < 54)'] = add

add = sumManualNighttime['Nighttime percentage in Hyperglycemia Level 1 (CGM < 70)'] 
sumManualNighttime['Nighttime percentage in HyperL1 Count'] = add

add = sumManualNighttime['Nighttime percentage in Secondary Range 70 <= CGM <= 150'] 
sumManualNighttime['Nighttime percentage in Secondary Range 70 <= CGM <= 150'] = add

add = sumManualNighttime['Nighttime percentage in First Range 70 <= CGM <= 180'] 
sumManualNighttime['Nighttime percentage in First Range 70 <= CGM <= 180'] = add

add = sumManualNighttime['Nighttime percentage in Hyperglycemia CGM > 180'] 
sumManualNighttime['Nighttime percentage in Hyperglycemia > 180'] = add

add = sumManualNighttime['Nighttime percentage in Critical Hyperglycemia CGM > 250']
sumManualNighttime['Nighttime percentage in Critical Hyperglycemia CGM > 250'] = add


# In[315]:


manual_nightly_avg = sumManualNighttime.mean() / 288 * 100
df = pd.DataFrame(manual_nightly_avg)
results = df.T.rename(index={0: "Manual Mode"})

results2 = results[['Nighttime percentage in Hyperglycemia CGM > 180', 
    "Nighttime percentage in Critical Hyperglycemia CGM > 250",
    'Nighttime percentage in First Range 70 <= CGM <= 180', 
    'Nighttime percentage in Secondary Range 70 <= CGM <= 150', 
    'Nighttime percentage in Hyperglycemia Level 1 (CGM < 70)', 
    'Nighttime percentage in Hyperglycemia Level 2 (CGM < 54)']]


# In[316]:


auto1 = cgm_auto.resample("D")["SensorGlucose"].value_counts(bins=threshold1, sort=False).to_frame(name="Whole Day percentage in Hyperglycemia Level 2 (CGM < 54)")
auto2 = cgm_auto.resample("D")['SensorGlucose'].value_counts(bins=threshold2, sort=False).to_frame(name="Whole Day percentage in Hyperglycemia Level 1 (CGM < 70)")
auto3 = cgm_auto.resample("D")['SensorGlucose'].value_counts(bins=threshold3, sort=False).to_frame(name="Whole Day percentage in secondary range 70 <= CGM <= 150")
auto4 = cgm_auto.resample("D")['SensorGlucose'].value_counts(bins=threshold4, sort=False).to_frame(name="Whole Day percentage in first range 70 <= CGM <= 180")
auto5 = cgm_auto.resample("D")["SensorGlucose"].value_counts(bins=threshold5, sort=False).to_frame(name="Whole Day percentage in Hyperglycemia CGM > 180")
auto6 = cgm_auto.resample("D")["SensorGlucose"].value_counts(bins=threshold6, sort=False).to_frame(name="Whole Day percentage in Critical Hyperglycemia CGM > 250")


# In[317]:


autoSum = pd.merge(auto1, auto2, on="Time", how="outer", sort=True)
autoSum = pd.merge(autoSum, auto3, on='Time', how='outer', sort=True)
autoSum = pd.merge(autoSum, auto4, on="Time", how="outer", sort=True)
autoSum = pd.merge(autoSum, auto5, on="Time", how="outer", sort=True)
autoSum = pd.merge(autoSum, auto6, on="Time", how='outer', sort=True)


# In[318]:


autoSum = autoSum.fillna(0)
autoDailySum = autoSum


# In[319]:


#calc percentages
add = autoDailySum['Whole Day percentage in Hyperglycemia Level 2 (CGM < 54)']
autoDailySum['Whole Day percentage in Hyperglycemia Level 2 (CGM < 54)'] = add

add = autoDailySum['Whole Day percentage in Hyperglycemia Level 1 (CGM < 70)']
autoDailySum['Whole Day percentage in Hyperglecemia Level 1 (CGM < 70)'] = add

add = autoDailySum['Whole Day percentage in secondary range 70 <= CGM <= 150']
autoDailySum['Whole Day percentage in secondary range 70 <= CGM <= 150)'] = add

add = autoDailySum['Whole Day percentage in first range 70 <= CGM <= 180']
autoDailySum['Whole Day percentage in first range 70 <= CGM <= 180'] = add

add = autoDailySum['Whole Day percentage in Hyperglycemia CGM > 180']
autoDailySum['Whole Day percentage in Hyperglycemia CGM > 180'] = add

add = autoDailySum["Whole Day percentage in Critical Hyperglycemia CGM > 250"]
autoDailySum['Whole Day percentage in Critical Hyperglycemia CGM > 250'] = add


# In[320]:


autoDailyAvg = autoDailySum.mean() / 288 * 100
df = pd.DataFrame(autoDailyAvg)
results = df.T.rename(index={0: "Auto Mode"})

results3 = results[['Whole Day percentage in Hyperglycemia CGM > 180', 
    "Whole Day percentage in Critical Hyperglycemia CGM > 250",
    'Whole Day percentage in first range 70 <= CGM <= 180', 
    'Whole Day percentage in secondary range 70 <= CGM <= 150', 
    'Whole Day percentage in Hyperglycemia Level 1 (CGM < 70)', 
    'Whole Day percentage in Hyperglycemia Level 2 (CGM < 54)']]


# In[321]:


day = cgm_auto.index.hour
mask = (day >= 6) & (day < 24)

auto1_day = cgm_auto[mask].groupby(cgm_auto[mask].index.floor('D'))["SensorGlucose"].value_counts(bins=threshold1, sort=False).to_frame(name="Daytime percentage in Hyperglycemia Level 2 (CGM < 54)")
auto2_day = cgm_auto[mask].groupby(cgm_auto[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold2, sort=False).to_frame(name="Daytime percentage in Hyperglycemia Level 1 (CGM < 70)")
auto3_day = cgm_auto[mask].groupby(cgm_auto[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold3, sort=False).to_frame(name="Daytime percentage in Secondary Range 70 <= CGM <= 150")
auto4_day = cgm_auto[mask].groupby(cgm_auto[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold4, sort=False).to_frame(name="Daytime percentage in First Range 70 <= CGM <= 180")
auto5_day = cgm_auto[mask].groupby(cgm_auto[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold5, sort=False).to_frame(name="Daytime percentage in Hyperglycemia CGM > 180")
auto6_day = cgm_auto[mask].groupby(cgm_auto[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold6, sort=False).to_frame(name="Daytime percentage in Critical Hyperglycemia CGM > 250")

autoSum = pd.merge(auto1_day, auto2_day, on="Time", how="outer", sort=True)
autoSum = pd.merge(autoSum, auto3_day, on='Time', how='outer', sort=True)
autoSum = pd.merge(autoSum, auto4_day, on="Time", how="outer", sort=True)
autoSum = pd.merge(autoSum, auto5_day, on="Time", how="outer", sort=True)
autoSum = pd.merge(autoSum, auto6_day, on="Time", how='outer', sort=True)

autoSum = autoSum.fillna(0)


# In[322]:


autoSum = autoSum[['Daytime percentage in Hyperglycemia Level 2 (CGM < 54)', 
    'Daytime percentage in Hyperglycemia Level 1 (CGM < 70)', 
    'Daytime percentage in Secondary Range 70 <= CGM <= 150', 
    'Daytime percentage in First Range 70 <= CGM <= 180', 
    'Daytime percentage in Hyperglycemia CGM > 180', 
    'Daytime percentage in Critical Hyperglycemia CGM > 250']]

autoSumDay = autoSum


# In[323]:


add = autoSumDay['Daytime percentage in Hyperglycemia Level 2 (CGM < 54)']
autoSumDay['Daytime percentage in Hyperglycemia Level 2 (CGM < 54)'] = add

add = autoSumDay['Daytime percentage in Hyperglycemia Level 1 (CGM < 70)'] 
autoSumDay['Daytime percentage in HyperL1 Count'] = add

add = autoSumDay['Daytime percentage in Secondary Range 70 <= CGM <= 150'] 
autoSumDay['Daytime percentage in Secondary Range 70 <= CGM <= 150'] = add

add = autoSumDay['Daytime percentage in First Range 70 <= CGM <= 180'] 
autoSumDay['Daytime percentage in First Range 70 <= CGM <= 180'] = add

add = autoSumDay['Daytime percentage in Hyperglycemia CGM > 180'] 
autoSumDay['Daytime percentage in Hyperglycemia > 180'] = add

add = autoSumDay['Daytime percentage in Critical Hyperglycemia CGM > 250']
autoSumDay['Daytime percentage in Critical Hyperglycemia CGM > 250'] = add


# In[324]:


autoDaySum = autoSumDay.mean() / 288 * 100
df = pd.DataFrame(autoDaySum)
results = df.T.rename(index={0: "Auto Mode"})

results4 = results[['Daytime percentage in Hyperglycemia CGM > 180', 
    "Daytime percentage in Critical Hyperglycemia CGM > 250",
    'Daytime percentage in First Range 70 <= CGM <= 180', 
    'Daytime percentage in Secondary Range 70 <= CGM <= 150', 
    'Daytime percentage in Hyperglycemia Level 1 (CGM < 70)', 
    'Daytime percentage in Hyperglycemia Level 2 (CGM < 54)']]


# In[325]:


night = cgm_auto.index.hour
mask = (night >= 0) & (night < 6)

auto1_night = cgm_auto[mask].groupby(cgm_auto[mask].index.floor('D'))["SensorGlucose"].value_counts(bins=threshold1, sort=False).to_frame(name="Nighttime percentage in Hyperglycemia Level 2 (CGM < 54)")
auto2_night = cgm_auto[mask].groupby(cgm_auto[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold2, sort=False).to_frame(name="Nighttime percentage in Hyperglycemia Level 1 (CGM < 70)")
auto3_night = cgm_auto[mask].groupby(cgm_auto[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold3, sort=False).to_frame(name="Nighttime percentage in Secondary Range 70 <= CGM <= 150")
auto4_night = cgm_auto[mask].groupby(cgm_auto[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold4, sort=False).to_frame(name="Nighttime percentage in First Range 70 <= CGM <= 180")
auto5_night = cgm_auto[mask].groupby(cgm_auto[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold5, sort=False).to_frame(name="Nighttime percentage in Hyperglycemia CGM > 180")
auto6_night = cgm_auto[mask].groupby(cgm_auto[mask].index.floor('D'))['SensorGlucose'].value_counts(bins=threshold6, sort=False).to_frame(name="Nighttime percentage in Critical Hyperglycemia CGM > 250")

autoSum = pd.merge(auto1_night, auto2_night, on="Time", how="outer", sort=True)
autoSum = pd.merge(autoSum, auto3_night, on='Time', how='outer', sort=True)
autoSum = pd.merge(autoSum, auto4_night, on="Time", how="outer", sort=True)
autoSum = pd.merge(autoSum, auto5_night, on="Time", how="outer", sort=True)
autoSum = pd.merge(autoSum, auto6_night, on="Time", how='outer', sort=True)

autoSum = autoSum.fillna(0)

autoSum = autoSum[['Nighttime percentage in Hyperglycemia Level 2 (CGM < 54)', 
    'Nighttime percentage in Hyperglycemia Level 1 (CGM < 70)', 
    'Nighttime percentage in Secondary Range 70 <= CGM <= 150', 
    'Nighttime percentage in First Range 70 <= CGM <= 180', 
    'Nighttime percentage in Hyperglycemia CGM > 180', 
    'Nighttime percentage in Critical Hyperglycemia CGM > 250']]

autoSumNight = autoSum


# In[326]:


add = autoSumNight['Nighttime percentage in Hyperglycemia Level 2 (CGM < 54)']
autoSumNight['Nighttime percentage in Hyperglycemia Level 2 (CGM < 54)'] = add

add = autoSumNight['Nighttime percentage in Hyperglycemia Level 1 (CGM < 70)']
autoSumNight['Nighttime percentage in HyperL1 Count'] = add

add = autoSumNight['Nighttime percentage in Secondary Range 70 <= CGM <= 150']
autoSumNight['Nighttime percentage in Secondary Range 70 <= CGM <= 150'] = add

add = autoSumNight['Nighttime percentage in First Range 70 <= CGM <= 180'] 
autoSumNight['Nighttime percentage in First Range 70 <= CGM <= 180'] = add

add = autoSumNight['Nighttime percentage in Hyperglycemia CGM > 180'] 
autoSumNight['Nighttime percentage in Hyperglycemia > 180'] = add

add = autoSumNight['Nighttime percentage in Critical Hyperglycemia CGM > 250']
autoSumNight['Nighttime percentage in Critical Hyperglycemia CGM > 250'] = add


# In[327]:


autoNightSum = autoSumNight.mean() / 288 * 100
df = pd.DataFrame(autoNightSum)
results = df.T.rename(index={0: "Auto Mode"})

results5 = results[['Nighttime percentage in Hyperglycemia CGM > 180', 
    "Nighttime percentage in Critical Hyperglycemia CGM > 250",
    'Nighttime percentage in First Range 70 <= CGM <= 180', 
    'Nighttime percentage in Secondary Range 70 <= CGM <= 150', 
    'Nighttime percentage in Hyperglycemia Level 1 (CGM < 70)', 
    'Nighttime percentage in Hyperglycemia Level 2 (CGM < 54)']]

manualData = [results2, results1, results0]
autoData = [results5, results4, results3]
manual = pd.concat(manualData, axis=1)
auto = pd.concat(autoData, axis=1)

rows = [manual, auto]
final = pd.concat(rows, axis=0)

final.to_csv("Miller_Results.csv")


# In[ ]:





# In[ ]:





# In[ ]:




