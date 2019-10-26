#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:46:18 2019

runs all three algorithms on a problem
this is non belina version with plotting and no saving

@author: Hook
"""

import numpy as np
import pickle
from Greedy_Greedy import Greedy_Greedy
from Entropy_Thompson import Entropy_Thompson
from Fixed_Thompson import Fixed_Thompson
    
iii=0
jjj=2
with open('problems/problem_data_'+str(iii)+'_'+str(jjj)+'.pkl', 'rb') as f:
    problem = pickle.load(f)
   
cz=0.3
cy=1-cz

H_FT=Fixed_Thompson(problem['data'],problem['hyperparameters'],cz,cy,ploton=True)
H_GG=Greedy_Greedy(problem['data'],problem['hyperparameters'],cz,cy,ploton=True)
H_ET=Entropy_Thompson(problem['data'],problem['hyperparameters'],cz,cy,ploton=True)

top=np.argsort(problem['data'][2])[-10:]

print(sum(1 for h in H_FT if h[0] in top and h[1]=='y'))
print(sum(1 for h in H_GG if h[0] in top and h[1]=='y'))
print(sum(1 for h in H_ET if h[0] in top and h[1]=='y'))

print(H_FT[-1])
print(H_GG[-1])
print(H_ET[-1])