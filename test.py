#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np


# In[3]:


#declaretions===================================================================================================================
normalization_coef = 9
batch_size = 10
kernel_size = 30
depth = 20
num_hidden = 100
num_channels = 2

learning_rate = 0.0001
training_epochs = 3

filter_value = 20


# In[4]:


data1 = pd.read_csv("acc1.csv") 
# Preview the first 5 lines of the loaded data
data2 = pd.read_csv("Acc2.csv")
#print(data1.head())
#============the frist log file (acc1) =========================================================================================
x = data1['X'].array
y = data1['Y'].array
z = data1['Z'].array
tt= data1['TT'].array
ang=data1['ANG'].array
#==============the sec log file (Acc2)==========================================================================================
x2 = data2['X'].array
y2 = data2['Y'].array
z2 = data2['Z'].array
tt2= data2['TT'].array
ang2=data2['ANG'].array
#===========cosim beetwen x=====================================================================================================


# In[5]:


#===========compare 1st file with 2end =========================================================================================
xs = dot(x, x2[:-1])/(norm(x)*norm(x2[:-1]))
print(xs)
#///////////////////////////////////////////
ys = dot(y, y2[:-1])/(norm(y)*norm(y2[:-1]))
print(ys)
#///////////////////////////////////////////
zs = dot(z, z2[:-1])/(norm(z)*norm(z2[:-1]))
print(zs)
#///////////////////////////////////////////
angs = dot(ang, ang2[:-1])/(norm(ang)*norm(ang2[:-1]))
print(angs)
#print(pd.DataFrame.equals(data1, data2))


# In[6]:


#====compare file one with it self =================================:new section:===============================================
xs = dot(x, x)/(norm(x)*norm(x))
print(xs)

ys = dot(y, y)/(norm(y)*norm(y))
print(ys)

zs = dot(z, z)/(norm(z)*norm(z))
print(zs)

angs = dot(ang, ang)/(norm(ang)*norm(ang))
print(angs)
#print(pd.DataFrame.equals(data1, data2))


# In[7]:


#non filtered datagrams=========================================================================================================
hi = data1.iloc[:, [1, 2,3]]
hi2= data2.iloc[:, [1, 2, 3]]
#print (hi)
#cos_lib = cosine_similarity(hi, hi2[:-1])
#print(cos_lib)
plt.plot(hi, hi, label='blot')
plt.legend()
plt.show()
plt.plot(hi, hi2[:-1])
plt.show()
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

plt.figure(1)
plt.subplot(311)
plt.plot(hi, f(hi), 'bo', hi2[:-1], f(hi2[:-1]), 'k')
plt.show()

plt.scatter(x, x2[:-1], marker='o');


# In[8]:


#===============compare the 1st with the 2end ==================================================================================
data3 = pd.read_csv("acc3.csv")
x3 = data3['X'].array
y3 = data3['Y'].array
z3 = data3['Z'].array
tt3= data3['TT'].array
ang3=data3['ANG'].array
#////////////////////////////////////////////
xs = dot(x, x3[:-3])/(norm(x)*norm(x3[:-2]))
print(xs)
#////////////////////////////////////////////
ys = dot(y, y3[:-3])/(norm(y)*norm(y3[:-3]))
print(ys)
#////////////////////////////////////////////
zs = dot(z, z3[:-3])/(norm(z)*norm(z3[:-3]))
print(zs)
#////////////////////////////////////////////
angs = dot(ang, ang3[:-3])/(norm(ang)*norm(ang3[:-3]))
print(angs)


# In[9]:


#compare the frist file with 4th with out filtring =============================================================================
data4 = pd.read_csv("acc4.csv")
x4 = data4['X'].array
y4 = data4['Y'].array
z4 = data2['Z'].array
tt4= data2['TT'].array
ang4=data2['ANG'].array
#////////////////////////////////////////////
xs = dot(x, x4[:-2])/(norm(x)*norm(x4[:-2]))
print(xs)
#///////////////////////////////////////////
ys = dot(y, y4[:-2])/(norm(y)*norm(y4[:-2]))
print(ys)
#///////////////////////////////////////////
zs = dot(z, z4[:-1])/(norm(z)*norm(z4[:-1]))
print(zs)
#///////////////////////////////////////////
angs = dot(ang, ang4[:-1])/(norm(ang)*norm(ang4[:-1]))
print(angs)


# In[38]:


#filter section================================:new section:====================================================================
def convolve1d(signal, length):
    ir = np.ones(length)/length
    #return np.convolve(y, ir, mode='same')
    
    output = np.zeros_like(signal)

    for i in range(len(signal)):
        for j in range(len(ir)):
            if i - j < 0: continue
            output[i] += signal[i - j] * ir[j]
            
    return output

def filterRecord(record, filter_value):
    x = convolve1d(record[:,0], filter_value)
    y = convolve1d(record[:,1], filter_value)
    z = convolve1d(record[:,2], filter_value)
    return np.dstack([x,y,z])[0]


def readFileData(file):
    column_names = ['TT', 'X', 'Y', 'Z']
    data = pd.read_csv("acc1.csv")
    
    x = data['X']
    y = data['Y']
    z = data['Z']
    
    return np.dstack([x,y,z])[0]

f1=readFileData("acc1.csv")
f2=readFileData("Acc2.csv")

print(f1)
print("=================================================================================")
print(f2)
print("=================================================================================")
c1=convolve1d(f1,605)
c2=convolve1d(f2,605)
print(c1)
print("=================================================================================")
print(c2)
print("=================================================================================")
filter1=filterRecord(c1,filter_value)
filter2=filterRecord(c2,filter_value)
print(cosine_similarity(filter1, filter2))
print("=================================================================================")
plt.scatter(filter2, filter1, marker='o');

plt.plot(filter1, filter2)
plt.show()

plt.figure(1)
plt.subplot(311)
plt.plot(filter1, f(filter1), 'or', filter2, f(filter2), 'k')
plt.show()


# In[42]:


#try to use cosine similarity after feltring==========:new section:=============================================================
#f1/////////////////////////////////////////
x1=filter1[:, 0]
y1=filter1[:, 1]
z1=filter1[:, 2]
#f2/////////////////////////////////////////
x2=filter2[:, 0]
y2=filter2[:, 1]
z2=filter2[:, 2]
#compare///////////////////////////////////
xo = dot(x1, x2)/(norm(x1)*norm(x2))
print(xo)
#////////////////////////////////////////////
yo = dot(y1, y2)/(norm(y1)*norm(y2))
print(yo)
#////////////////////////////////////////////
zo = dot(z1, z2)/(norm(z1)*norm(z2))
print(zo)
#////////////////////////////////////////////

