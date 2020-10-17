#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np
import math
import matplotlib.pyplot as plt
import random
import csv


# In[76]:


def toRange(arr, min, max):
    return arr*(max-min)+min;

def randInRange(min, max, shape = 1):
    num = np.random.random(shape)
    return num*(max-min)+min;


# In[77]:


def readCsvNpArray(filename):
    data = np.genfromtxt(filename, delimiter=';')
    return data


# In[78]:


trainSet = readCsvNpArray('Tabela_RNA_TREINO.csv')
trainSet


# In[79]:


nIns = trainSet.shape[1]-1
nIns

targets = np.unique(trainSet[:,nIns])
targets

nOuts = targets.shape[0]
nOuts


# In[80]:


# 
# plt.plot(self.weights[:,0],self.weights[:,1], color='black')
# plt.title("W1 x W2")
# plt.plot(inputs[:,0], inputs[:,1], 'o', color='black');
# plt.show()


# In[81]:


def getTarget(arr):
    if(len(arr.shape) == 2):
        return arr[:,arr.shape[1]-1]
    return arr[arr.shape[0]-1]

def getInputs(arr):
    if(len(arr.shape) == 2):
        return arr[:,0:arr.shape[1]-1]
    else:
        return arr[0:arr.shape[0]-1]


# In[82]:


ts = getTarget(trainSet)
ts


# In[83]:


colors = {
    1:'blue',
    2:'darkred',
    3:'green',
    4:'violet'
}
centroideColors = {
    1:'cyan',
    2:'red',
    3:'limegreen',
    4:'deeppink'
}
def getColor(target):
    return colors[target]

getColorNpArray = np.vectorize(getColor)


# In[84]:


class DataClass(object):

    def __init__(self, target, classData):
        self.data = classData
        self.inputs = getInputs(classData)
        self.target = target
        self.calcMeans()
        
    def calcMeans(self):
        self.means = getInputs(self.data).mean(0)


# In[85]:


def clusterClasses(mtx, targets):
    classes = {}
    for t in targets:
        classes[t] = []
    
    for inputArr in mtx:
        t = getTarget(inputArr)
        classes[t].append(inputArr)
        
    for t in targets:
        classes[t] = DataClass(t, np.array(classes[t]))
#     print(classes[1.0].data)
    return classes


# In[86]:


# getTarget(trainSet[0])
# trainSet[0]
# getInputs(trainSet[0])


# In[87]:


classes = clusterClasses(trainSet, targets)
# classes
classes[2].data


# In[88]:


trainSet


# In[147]:


class LVQ(object):

    def __init__(self, nIns, nOuts, learningRate, maxEpoch):
        self.nIns = nIns
        self.nOuts = nOuts
        # matriz de pesos tranposta.
#         self.weights = toRange(np.random.rand(nOuts, nIns),1,-1)
        self.learningRate = learningRate
        self.maxEpoch = maxEpoch
        
    def initWeights(self, classes):
        self.weights = np.empty([nOuts, nIns])
        for i in range(self.nOuts):
            self.weights[i] = classes[i+1].means.copy()
        self.weights = toRange(np.random.rand(nOuts, nIns),6,0)
        
    def calcDist(self, x, j):
        wj = self.weights[j]
        distances = (wj-x)**2
        total_dist = np.sum(distances)
        return total_dist
        
    def getNearestUnit(self, x):
        jmin = 0
        distMin = self.calcDist(x, jmin)
        
        for j in range(self.nOuts):
            dist = self.calcDist(x, j)
            if dist < distMin:
                distMin = dist
                jmin = j
        return jmin
        
    def train(self, inputsAndT, targets, classes):
        self.epoch = 0
        self.targets = targets
        if self.epoch == 0:
                self.showGraphs(classes)
        
        while self.epoch <= self.maxEpoch:
            np.random.shuffle(inputsAndT)
            for xAndT in inputsAndT:
                x = getInputs(xAndT)
#                 print('x')
#                 print(x)
                tx = getTarget(xAndT)
                jmin = self.getNearestUnit(x)
                self.updateWeights(jmin,x, tx)
                    
            self.updateLearningRate()
            if self.epoch%10 == 0:
                self.showGraphs(classes)
            self.epoch+=1
            
    def classify(self, inputs):
        predictedTargets = np.empty(inputs.shape[0])
        
        for i in range(len(inputs)):
            x = inputs[i]
            jmin = self.getNearestUnit(x)
            predictedTargets[i] = self.targets[jmin]
        return predictedTargets
    
    def classifyAndCompleteMtx(self, inputs):
        predictedTargets = self.classify(inputs)
        completeMtx = np.c_[testSet,predictedTargets]
        predictedClasses = clusterClasses(completeMtx, targets)
        self.showGraphs(predictedClasses)
        return completeMtx
        
             
    def updateWeights(self, j, x, tx):
        wOld = self.weights[j]
        t = self.targets[j]
        # se target da unidade/neurônio for o mesmo do target da entrada tx
#         print('tx '+ str(tx) + ' t '+ str(t))
        if t == tx:
#             print('Igual')
            self.weights[j] = wOld + self.learningRate*(x - wOld)
        else:
#             print('Dif')
            self.weights[j] = wOld - self.learningRate*(x - wOld)
    
    def updateLearningRate(self):
#         print('learningRate')
#         print(self.learningRate)
#         self.learningRate = self.learningRate - 0.0049
        self.learningRate = self.learningRate * 0.96163508
#         self.learningRate = self.learningRate - 0.00049
        
    def showGraphs(self, classes):
        print('Epoch')
        print(self.epoch)        
        for i in range(len(self.targets)):
            t = self.targets[i]
            print('t '+ str(t))
#             print('self.weights ')
#             print(self.weights[i,:])
#             print('classes[t].inputs ')
#             print(classes[t].inputs)
            plt.plot(self.weights[i,:], color=centroideColors[t])
            inputs = classes[t].inputs
            for inp in inputs:
                plt.plot(inp, color=colors[t], linestyle='dashed');
                
            plt.title("Target "+ str(t) + " - X1 a X6 ") 
        plt.show()


# In[152]:


lvq  =  LVQ(nIns, nOuts, 0.3, 100)
lvq.initWeights(classes)
lvq.train(trainSet, targets, classes)
# lvq.weights


# In[153]:


testSet = readCsvNpArray('Tabela_RNA_TESTE.csv')
testSet


# In[154]:


predictedTargets = lvq.classify(testSet)
predictedTargets


# In[157]:


testSetWithTargets = lvq.classifyAndCompleteMtx(testSet)


# In[158]:


testSetWithTargets


# In[ ]:




