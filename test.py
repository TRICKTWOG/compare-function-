# Load the Pandas libraries with alias 'pd' 
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go


# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data1 = pd.read_csv("acc1.csv") 
# Preview the first 5 lines of the loaded data
data2 = pd.read_csv("Acc2.csv")
#print(data1.head())
#============the frist log file (acc1) ==========================
x = data1['X'].array
y = data1['Y'].array
z = data1['Z'].array
tt= data1['TT'].array
ang=data1['ANG'].array
#==============the sec log file (Acc2)============================
x2 = data2['X'].array
y2 = data2['Y'].array
z2 = data2['Z'].array
tt2= data2['TT'].array
ang2=data2['ANG'].array
#===========cosim beetwen x========================================
xs = dot(x, x2[:-1])/(norm(x)*norm(x2[:-1]))
print(xs)
#==========cosim beetwen y=========================================
ys = dot(y, y2[:-1])/(norm(y)*norm(y2[:-1]))
print(ys)
#=========cosim beetwen zs=========================================
zs = dot(z, z2[:-1])/(norm(z)*norm(z2[:-1]))
print(zs)
#=========cosim beetwen angs=======================================
angs = dot(ang, ang2[:-1])/(norm(ang)*norm(ang2[:-1]))
print(angs)
#print(pd.DataFrame.equals(data1, data2))

#================ blots ===========================================

import numpy as np
hi = data1.iloc[:, [1, 2,3]]
hi2= data2.iloc[:, [1, 2, 3]]
#print (hi)
cos_lib = cosine_similarity(hi, hi2[:-1])
print(cos_lib)
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
plt.subplot(212)
plt.show()

plt.scatter(x, x2[:-1], marker='o');
plt.show()
#==================================================================
print("============================================================")
#===============the 3th log file (acc3)============================
data1 = pd.read_csv("acc3.csv")
#==================================================================
x3 = data2['X'].array
y3 = data2['Y'].array
z3 = data2['Z'].array
tt3= data2['TT'].array
ang3=data2['ANG'].array
#===========the bettwen 1 and 3===================================
xs = dot(x, x3[:-1])/(norm(x)*norm(x3[:-1]))
print(xs)
#==========cosim beetwen y=========================================
ys = dot(y, y3[:-1])/(norm(y)*norm(y3[:-1]))
print(ys)
#=========cosim beetwen zs=========================================
zs = dot(z, z3[:-1])/(norm(z)*norm(z3[:-1]))
print(zs)
#=========cosim beetwen angs=======================================
angs = dot(ang, ang3[:-1])/(norm(ang)*norm(ang3[:-1]))
print(angs)
#print(pd.DataFrame.equals(data1, data2))
#==================the 4th ========================================
print("=============================================================")
#===============the 3th log file (acc3)============================
data4 = pd.read_csv("acc4.csv")
#==================================================================
x4 = data4['X'].array
y4 = data4['Y'].array
z4 = data4['Z'].array
tt4= data4['TT'].array
ang4=data4['ANG'].array
#===========the bettwen 1 and 3===================================
xs = dot(x, x4[:-1])/(norm(x)*norm(x4[:-1]))
print(xs)
#==========cosim beetwen y=========================================
ys = dot(y, y4[:-1])/(norm(y)*norm(y4[:-1]))
print(ys)
#=========cosim beetwen zs=========================================
zs = dot(z, z4[:-1])/(norm(z)*norm(z4[:-1]))
print(zs)
#=========cosim beetwen angs=======================================
angs = dot(ang, ang4[:-1])/(norm(ang)*norm(ang4[:-1]))
print(angs)
#print(pd.DataFrame.equals(data1, data2))
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////
#///////////////////////the diff betwen 2 && 3//////////////////////////////////////////////////////////
print("=================================================================")
xs = dot(x4[:-1], x2[:-1])/(norm(x4[:-1])*norm(x2[:-1]))
print(xs)
#==========cosim beetwen y=========================================
ys = dot(y4[:-1], y2[:-1])/(norm(y4[:-1])*norm(y2[:-1]))
print(ys)
#=========cosim beetwen zs=========================================
zs = dot(z4[:-1], z2[:-1])/(norm(z4[:-1])*norm(z2[:-1]))
print(zs)
#=========cosim beetwen angs=======================================
angs = dot(ang4[-1], ang2[:-1])/(norm(ang4[:-1])*norm(ang2[:-1]))
print(angs)
