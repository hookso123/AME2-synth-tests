#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:15:01 2019

runs all three algorithms on each problem
this is belina version 

this version for fixed costs

@author: Hook
"""

import numpy as np
import pickle
from Greedy_Greedy import Greedy_Greedy
from Entropy_Thompson import Entropy_Thompson
from Fixed_Thompson import Fixed_Thompson
import sys
    
def go(kkk):
    
    jjj=int(np.mod(kkk,5))
    iii=int((kkk-jjj)/5)
    
    with open('problems/problem_data_'+str(iii)+'_'+str(jjj)+'.pkl', 'rb') as f:
        problem = pickle.load(f)
       
    cz=0.3
    cy=1-cz
    
    H_FT=Fixed_Thompson(problem['data'],problem['hyperparameters'],cz,cy,ploton=True)
    H_GG=Greedy_Greedy(problem['data'],problem['hyperparameters'],cz,cy,ploton=True)
    H_ET=Entropy_Thompson(problem['data'],problem['hyperparameters'],cz,cy,ploton=True)
    
    with open('results_fixed_costs/results_'+str(iii)+'_'+str(jjj)+'.pkl', 'wb') as f:
        pickle.dump({'H_FT':H_FT,'H_GG':H_GG,'H_ET':H_ET}, f)
        
if __name__ == "__main__":
    go(int(sys.argv[1]))  
