#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
class CMA:
    def __init__(self,startMean, n):
        #mean is a real, no vector,  n= number of variable in vector
        self.LAMBDA = 4+3*np.log(n) # offspring number, new solutions sampled, population size
        self.N = n                  # variables in each vector
        self.MEW = self.LAMBDA/2        # μ, parent number, solutions involved in updates of m, C, and sigma
        self.M = np.zeros(n)

        self.C = np.identity(n)     
        self.CC = 4/n
        self.CCOV = (2 + self.MEWCOV**2)/(self.N**2)    #learning rate for C-update
        self.CSIG = 4/n
        self.C1 = 2/(n**2)
        self.CMEW = self.MEWW/(n**2) # intenionally mew-W not a typo
        self.DSIG = 1 + np.sqrt(self.MEWW / n)

        self.WPRIME = np.array(int(self.LAMBDA))
        for i in range(int(self.LAMBDA)):
            self.WPRIME[i] = np.log()
        
        
   
    
        self.PSIG = np.zeros(self.N)
        self.PC = np.zeros(self.N)
        self.eig = np.linalg.eigh(self.C)
        self.E = sqrt(n)*(1-1/(4*n)+ 1/(21*n**2))
        self.SIG = .5 #num greater than zero  EXPERIMENT by changing this
        self.MEAN = startMean   
        self.DSIG = 1 + self.MEWCOV/sqrt(self.N)
        
        self.WEIGHTS = np.zeros(int(self.MEW))
        count = 0
        for i in self.WEIGHTS:
            count+=1
            i = np.log(self.MEW + .5) + np.log(count)
        summation = 0
        for i in self.WEIGHTS:
            summation += i**2
        self.MEWCOV = np.sqrt(1/summation)
        self.Z = np.zeros((int(self.LAMBDA),self.N))
        for row in self.Z:
            row = np.random.multivariate_normal(self.M,np.identity(self.N))


        self.ZW = np.zeros(self.N)


        for wi in self.W:
            self.ZW += wi*self.Z


    
    
    
    # random sampled values = NORMALVIARAITE DIST(0,C) == B @ D @ NORMALVARIATE DIST(0,I)
    
    def changePC(self):
        self.PC = (1-self.CC)*self.PC + sqrt(self.CC*(2-self.CC))*self.MEWCOV*self.Z
        self.ZSUM = 0
        for i in range(1,self.MEW+1):
            self.ZSUM += WEIGHT[i]*self.ZLAMBDA @ self.ZLAMBDA.T
        self.C = 1-self.CCOV)*self.C + self.CCOV/self.MEWCOV* self.PC @ self.PC.T + CCOV*(1-1/self.MEWCOV)*self.ZSUM
    def stepPC(self):
        self.PC = (1-self.LR)*self.PC + self.LR*(<z>)
    def stepC():
        self.C = (1-self.Ccov)*self.C + self.CCOV/self.MEWCOV* self.PC @(self.PC.T)
    def stepPSIG(self):
        (values,vectors) = np.linalg.eigh(self.C) 
        rootInvC = vectors / np.det(vectors)
        # need to add more stuff to make it properly get the eigenvalues and get root
        self.PSIG = (1-self.CSIG)*self.PSIG + np.sqrt(1-(1-self.CSIG)**2) * self.MEWCOV * rootInvC @ self.ZW
    def stepZ(self):
        self.Z = np.random.multivariate_normal(self.Z,self.C)
    def stepSIG(self):
        self.SIG = self.SIG*e^(self.CSIG/self.DSIG * (np.linalg.norm(self.PSIG)/np.linalg.norm(np.random.multivariate_normal(self.M,np.identity(self.N)))/self.E - 1))
    
    
    def plot(self):
        import pylab as plt
        import seaborn as sns

        sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})

        get_ipython().run_line_magic('matplotlib', 'inline')
        plt.figure(figsize=(12,12))
        plt.plot(self.BEST_OUTPUTS,'-o', lw=3, ms=20, label='from scratch')
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.show()
    

class Knapsack(CMA):
    def __init__(self, n, item_values, item_weights, bag_capacity):
        initialVector = np.random.multivariate_normal(np.zeros(n),np.identity(n))
        super().__init__(initialVector,n)
        self.ITEM_VALUES = item_values
        self.ITEM_WEIGHTS = item_weights
        self.BAG = bag_capacity


    def fitness(self, vector):
        totalWeight = vector*self.ITEM_WEIGHTS
        if totalWeight > self.BAG:
            return 0
        return vector * self.ITEM_VALUES

class Parabola(CMA):
    def __init__(self):
        initialVector = np.random.multivariate_normal(np.zeros(1),np.identity(1))
        super().__init__(initialVector,1)

    def fitness(self, vector):
        return vector * vector * -10

class HyperEllipsoid(CMA):
    def __init__(self):
        initialVector = np.random.multivariate_normal(np.zeros(2),np.identity(2))
        super().__init__(initialVector,2)

    def fitness(self, vector):
        x = vector[0]
        x2 = vector[1]
        return  -((np.sqrt(3)/5*(x-3) + .5*(x2-5))**2 + 5*(np.sqrt(3)/5*(x-3) + .5*(x2-5))**2)

class Rastrigin(CMA):

    def __init__(self, n):
        initialVector = np.random.multivariate_normal(np.zeros(n),np.identity(n))
        super().__init__()

    def fitness(self):
        count = 10* self.N
        for i in range(0,self.N):
            count+= (vector[i]-2)**2 - 10*np.cos(2*np.pi*(vector[i]-2))
        return -count
            

