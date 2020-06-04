#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:49:59 2020

@author: usama
"""


from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import matplotlib.pyplot as plt
import neal
import numpy as np
from numpy.random import randn

token='Enter Dwave Token'
sampler = EmbeddingComposite(DWaveSampler(token=token,solver={'qpu': True}))

#sampler=neal.SimulatedAnnealingSampler()

#This function is taken from Dr Fayyaz ul Amir Afsar Minhas (Github User: foxtrotmike)
def getExamples(n=100,d=2):
    """
    Generates n d-dimensional normally distributed examples of each class        
    The mean of the positive class is [1] and for the negative class it is [-1]
    DO NOT CHANGE THIS FUNCTION
    """
    Xp = randn(n,d)+1   #generate n examples of the positive class
    #Xp[:,0]=Xp[:,0]+1
    Xn = randn(n,d)-1   #generate n examples of the negative class
    #Xn[:,0]=Xn[:,0]-1
    X = np.vstack((Xp,Xn))  #Stack the examples together to a single matrix
    Y = np.array([+1]*n+[-1]*n) #Associate Labels
    return X,Y

def getMedianNonZero(data):
    data=data.reshape(1,-1)[0]
    x=np.nonzero(data)
    data=data[x]
    return np.median(data)

data,actualLabel=getExamples(n=10,d=2)

sh=data.shape[0]
mat=np.zeros([sh,sh])

#Creating sparse matrix based on distance # 
for i in range(len(data)):
    for j in range(i):
        x=data[i]
        y=data[j]
        mat[i,j] = np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))


mat=np.matrix(mat)       
mat=mat.T
mat=np.array(mat)

# Calculating median of nonzero values of matrix, if value is less than median then setting distance to -1 else nothing.

medVal=getMedianNonZero(mat)
tt=mat.reshape(1,-1)[0]

l=[]
for i in tt:
    if i<medVal:
        l.append(i*-1)
    else:
        l.append(i)

l=np.array(l)   
mat=l.reshape(sh,sh)

# Sending the matrix to sampler
output=sampler.sample_qubo(mat)
classes=output.record[0][0]
print(output.record[0])

# Plotting
plt.close('all')
idx=0
for i in data:
    if classes[idx]==0:
        plt.scatter(i[0],i[1],c='b')
    else:
        plt.scatter(i[0],i[1],c='r')
    idx=idx+1
