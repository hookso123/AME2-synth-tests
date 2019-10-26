#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 09:35:14 2019

creates a set of test problems for the two stage synth algorithms

@author: Hook
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import pickle

ploton=True

for iii in range(2):
    for jjj in range(5):

        n=500
        x=np.random.uniform(0,1,n)
        Dx=euclidean_distances(x.reshape(-1,1),x.reshape(-1,1),squared=True)
        
        az=0.25
        bz=0.25
        lz=0.25
        
        SIG_z=az**2*np.exp(-Dx/(2*lz**2))+bz**2*np.identity(n)
        z=np.random.multivariate_normal(np.zeros(n),SIG_z)
        
        Dz=euclidean_distances(z.reshape(-1,1),z.reshape(-1,1),squared=True)
        
        ay=1
        by=0.05
        THETA=np.linspace(0,np.pi/2,5)
        theta=THETA[jjj]
        lyx=2**2.5*np.sin(theta)
        lyz=2**2.5*np.cos(theta)
        
        SIG_y=ay**2*np.exp(-Dx*lyx**2/2-Dz*lyz**2/2)+by**2*np.identity(n)
        y=np.random.multivariate_normal(np.zeros(n),SIG_y)
        
        N=10
        
        with open('problems/problem_data_'+str(iii)+'_'+str(jjj)+'.pkl', 'wb') as f:
            pickle.dump({'data':[x,z,y],'hyperparameters':[az,bz,lz,ay,by,theta,lyx,lyz]}, f)
        
        top=np.argsort(y)[-N:]
        if ploton:
            plt.plot(x[top],z[top],'.',color='red')
            plt.scatter(x,z,10,y)
            plt.show()