#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
class CMA:
    def __init__(self,mean, n):
        #mean is a real, no vector,  n= number of variable in vector
        self.LAMBDA = 4+3*np.log(n)
        self.N = n
        self.C = np.identity(n)
        self.CC = 4/n
        self.CCOV = (2 + self.MEWCOV**2)/(self.N**2)
        self.CSIG = 4/n
        self.C1 = 2/(n**2)
        self.WPRIME = np.array(int(self.LAMBDA))
        for i in range(int(self.LAMBDA)):
            self.WPRIME[i] = np.log()
        
        
   
    
    self.PSIG = np.zeros(self.N)
    self.PC = np.zeros(self.N)
    self.eig = np.linalg.eigh(self.C)
    self.E = sqrt(n)*(1-1/(4*n)+ 1/(21*n**2))
    self.SIG = #num greater than zero
    self.MEAN = np.zeros(self.N)# INITALLY zero vector of 
    self.MEW = self.LAMBDA/2
    self.DSIG = 1 + self.MEWCOV/sqrt(self.N)
    
    self.WEIGHTS = np.array(int(self.MEW))
    count = 0
    for i in self.WEIGHTS:
        count+=1
        i = np.log(self.MEW + .5) + np.log(count)
    sum = 0
    for i in self.WEIGHTS:
        sum += i**2
    self.MEWCOV = np.sqrt(1/sum)
    
    
    
    # random sampled values = NORMALVIARAITE DIST(0,C) == B @ D @ NORMALVARIATE DIST(0,I)
    
    def changePC():
        self.PC = (1-self.CC)*self.PC + sqrt(self.CC*(2-self.CC))*self.MEWCOV*self.Z
        self.ZSUM = 0
        for i in range(1,self.MEW+1):
            self.ZSUM += WEIGHT[i]*self.ZLAMBDA @ self.ZLAMBDA.T
        self.C = 1-self.CCOV)*self.C + self.CCOV/self.MEWCOV* self.PC @ self.PC.T + CCOV*(1-1/self.MEWCOV)*self.ZSUM
    def stepPC():
        self.PC = (1-self.LR)*self.PC + self.LR*(<z>)
    def stepC():
        self.C = (1-self.Ccov)*self.C + self.CCOV/self.MEWCOV* self.PC @(self.PC.T)
    def stepPSIG():
        self.PSIG = (1-self.CSIG)*self.PSIG + np.sqrt()
class Rastrigin(CMA):
    def __init__(self, n):
        self.N = n
    def fitness(self,vector):
        count = 10* self.N
        for i in range(0,self.N):
            count+= (vector[i]-2)**2 - 10*np.cos(2*np.pi*(vector[i]-2))
        return count
            

