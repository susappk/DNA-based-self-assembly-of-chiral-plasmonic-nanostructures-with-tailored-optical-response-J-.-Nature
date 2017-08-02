#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 05:50:12 2017

@author: xiaabs
"""
import numpy as np
def wavepeak(a,lam,mode):
    b=[]
    for i in range(len(a)-2):
        i=1+i      
        if (a[i]>a[i+1])and(a[i]>a[i-1])and(i>30)and(i<970)and(a[i]==np.max(a[i-20:i+20])): 
            if mode=="str":
                b.append((format(lam[i],'.1f'),format(a[i],'.2e')))
            elif mode =='number':
                b.append(a[i])
                b[0]=np.max(b)
    return b
def wavedeep(a,lam,mode):
    b=[]
    for i in range(len(a)-2): 
        i=i+1
        if (a[i]<a[i+1])and(a[i]<a[i-1])and(i>30)and(i<970)and(a[i]==np.min(a[i-20:i+20])):
            if mode=="str":
                b.append((format(lam[i],'.1f'),format(a[i],'.2e')))
            elif mode =='number':
                b.append(a[i])
                b[0]=np.min(b)
    return b  