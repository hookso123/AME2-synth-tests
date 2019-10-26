#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:09:51 2019

Entropy controller with Thompson sampling 

@author: Hook
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:45:28 2019
same but with inverse length scales 
@author: jlh75
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import norm

def Entropy_Thompson(data,hyperparameters,cz,cy,B=50,N=10,ploton=False):
    
    x,z,y=data
    az,bz,lz,ay,by,theta,lyx,lyz=hyperparameters
    top=np.argsort(y)[-N:]
    n=x.shape[0]
    Dx=euclidean_distances(x.reshape(-1,1),x.reshape(-1,1),squared=True)  
    SIG_z=az**2*np.exp(-Dx/(2*lz**2))+bz**2*np.identity(n)  
    Dz=euclidean_distances(z.reshape(-1,1),z.reshape(-1,1),squared=True)  
    SIG_y=ay**2*np.exp(-Dx*lyx**2/2-Dz*lyz**2/2)+by**2*np.identity(n)   
    
    """ samples from posterior """
    def samples(tt,tu,uu,nz,nzy,ztttu,ytt):
        mu_cz=np.matmul(SIG_z[np.ix_(uu,tt+tu)],np.linalg.solve(SIG_z[np.ix_(tt+tu,tt+tu)],ztttu))
        SIG_cz=SIG_z[np.ix_(uu,uu)]-np.matmul(SIG_z[np.ix_(uu,tt+tu)],np.linalg.solve(SIG_z[np.ix_(tt+tu,tt+tu)],SIG_z[np.ix_(tt+tu,uu)]))
        ZS=np.zeros((n,nz))
        ZS[tt+tu,:]=np.repeat(ztttu.reshape(-1,1),nz,1)
        ZS[uu,:]=np.random.multivariate_normal(mu_cz,SIG_cz,nz).T
        YS=np.zeros((n,nz*nzy))
        YS[tt,:]=np.repeat(ytt.reshape(-1,1),nz*nzy,1)
        for i in range(nz):
            sampled_SIG_y=ay**2*np.exp(-Dx*lyx**2/2-euclidean_distances(ZS[:,i].reshape(-1,1),ZS[:,i].reshape(-1,1),squared=True)*lyz**2/2)+by**2*np.identity(n)
            sampled_mu_cy=np.matmul(sampled_SIG_y[np.ix_(tu+uu,tt)],np.linalg.solve(sampled_SIG_y[np.ix_(tt,tt)],ytt))
            sampled_SIG_cy=sampled_SIG_y[np.ix_(tu+uu,tu+uu)]-np.matmul(sampled_SIG_y[np.ix_(tu+uu,tt)],np.linalg.solve(sampled_SIG_y[np.ix_(tt,tt)],sampled_SIG_y[np.ix_(tt,tu+uu)]))
            YS[tu+uu,i*nzy:(i+1)*nzy]=np.random.multivariate_normal(sampled_mu_cy,sampled_SIG_cy,nzy).T
        return ZS,YS
    
    
    def entropy(x):
        I=[i for i in range(len(x)) if x[i]>0 and x[i]<1]
        H=-np.multiply(x[I],np.log(x[I]))
        h=sum(H)
        return h
    
    r=1
    tt=list(range(r))
    tu=[]
    uu=list(range(r,n))
    
    b=B
    History=[]
    res=5
    ZR=np.linspace(-2,2,res)
    PR=norm.pdf(ZR)
    PR=PR/np.sum(PR)
    while b>cy:
        """ sample from posterior to estimate greedy-N acquisituion function """
        ZS,alpha=samples(tt,tu,uu,2,1,z[tt+tu],y[tt])
        if len(tu)>0:
            """ select best candidates from TU and UU """
            itu=tu[np.argmax(alpha[tu,0])]
            iuu=uu[np.argmax(alpha[uu,1])]
            """ integrate over zuu to estimate profit of action z """
            mu_uuz=np.matmul(SIG_z[iuu,tt+tu],np.linalg.solve(SIG_z[np.ix_(tt+tu,tt+tu)],z[tt+tu]))
            SIG_uuz=SIG_z[iuu,iuu]-np.matmul(SIG_z[iuu,tt+tu],np.linalg.solve(SIG_z[np.ix_(tt+tu,tt+tu)],SIG_z[tt+tu,iuu]))
            mu_tuy=np.matmul(SIG_y[itu,tt],np.linalg.solve(SIG_y[np.ix_(tt,tt)],y[tt]))
            SIG_tuy=SIG_y[itu,itu]-np.matmul(SIG_y[itu,tt],np.linalg.solve(SIG_y[np.ix_(tt,tt)],SIG_y[tt,iuu]))
            ZuuR=mu_uuz+ZR*SIG_uuz**0.5
            YtuR=mu_tuy+ZR*SIG_tuy**0.5
            uuc=uu.copy()
            uuc.remove(iuu)
            tuc=tu.copy()
            tuc.remove(itu)
            tuc.append(iuu)
            ttc=tt.copy()
            ttc.append(itu)
            mz=5
            my=100
            P=np.zeros((n,res,res))
            for i in range(res):
                for j in range(res):
                    ZS,YS=samples(ttc,tuc,uuc,mz,100,np.concatenate((z[tt+tu],ZuuR[i].reshape(1))),np.concatenate((y[tt],YtuR[j].reshape(1))))
                    for k in range(mz*my):
                        I=[np.argpartition(YS[:,k],-N)[-N:]]
                        P[I,i,j]+=1
            P=P/(mz*my)
            p=np.zeros(n)
            for i in range(res):
                for j in range(res):
                    p=p+PR[j]*PR[i]*P[:,i,j]
            H=entropy(p)
            Hz=np.zeros(res)
            for i in range(res):
                p=np.zeros(n)
                for j in range(res):
                    p=p+PR[j]*P[:,i,j]
                Hz[i]=entropy(p)
            Hy=np.zeros(res)
            for j in range(res):
                p=np.zeros(n)
                for i in range(res):
                    p=p+PR[i]*P[:,i,j]
                Hy[j]=entropy(p)
                    
            DEz=np.dot(H-Hz,PR)
            DEy=np.dot(H-Hy,PR)
    
            if ploton:
                plt.figure(figsize=(15,5))
                plt.subplot(121)
                plt.plot(x[top],z[top],'.',color='red')
                plt.plot(x[tu],z[tu],'s',color='black')
                plt.plot(x[tt],z[tt],'d',color='red')
                plt.plot([x[iuu],x[iuu]],[np.min(z),np.max(z)],color='red')
                plt.plot(x[iuu],z[iuu],'x',color='red')
                plt.plot(x[itu],z[itu],'x',markersize=25,color='red')
                plt.scatter(x,z,10,y)
                plt.xlabel('x')
                plt.ylabel('z')
                plt.subplot(122)
                plt.plot(ZuuR,Hz,label='z')
                plt.plot([ZuuR[0],ZuuR[-1]],[H,H])
                plt.plot(YtuR,Hy,label='y')
                plt.plot([YtuR[0],YtuR[-1]],[H,H])
                plt.legend()
                plt.xlabel('z/y')
                plt.ylabel('expected entropy')
                plt.show()
            """ reward/cost ratio of different actions """
            alphay=DEy/cy
            alphaz=DEz/cz
            if alphay>alphaz:
                tu.remove(itu)
                tt.append(itu)
                b=b-cy
                History.append([itu,'y'])
            else:
                uu.remove(iuu)
                tu.append(iuu)
                b=b-cz
                History.append([iuu,'z'])
        else:
                iuu=uu[np.argmax(alpha[uu,0])]
                uu.remove(iuu)
                tu.append(iuu)
                b=b-cz  
                History.append([iuu,'z'])
        if ploton:
            plt.figure(figsize=(15,5))
            plt.subplot(121)        
            plt.plot(x[top],z[top],'.',color='red')
            plt.plot(x[tu],z[tu],'s',color='black')
            plt.plot(x[tt],z[tt],'d',color='red')
            if History[-1][1]=='y':
                 plt.plot(x[tt[-1]],z[tt[-1]],'d',color='red',markersize=15)
            else:
                plt.plot(x[tu[-1]],z[tu[-1]],'s',color='black',linewidth=3,markersize=15)
            plt.scatter(x,z,10,y)
            plt.xlabel('x')
            plt.ylabel('z')
            plt.subplot(122)
            if History[-1][1]=='y':
                 plt.plot(x[tt[-1]],z[tt[-1]],'d',color='red',markersize=15)
            else:
                plt.plot(x[tu[-1]],z[tu[-1]],'s',color='black',markersize=15)
            plt.scatter(x[tt],z[tt],50,y[tt])
            plt.plot(x[tu],z[tu],'s',color='black')
            plt.plot(x[uu],np.ones(len(uu))*np.min(z),'x')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.show()
            print(b)

    ZS,YS=samples(tt,tu,uu,10,100,z[tt+tu],y[tt])
    P=np.zeros(n)
    for i in range(1000):
        I=[np.argpartition(YS[:,i],-N)[-N:]]
        P[I]+=1
    P=P/1000
    H=entropy(P)
    History.append([H,'final_entropy'])

    return History
