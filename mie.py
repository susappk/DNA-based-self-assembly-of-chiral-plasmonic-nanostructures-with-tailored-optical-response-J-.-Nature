#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:16:30 2017

@author: xiaabs
"""
import numpy as np
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
import statsmodels.api as sm


wavelength_nm,exp_n,exp_k,fdtd_n,fdtd_k=np.loadtxt("/home/xiaabs/Desktop/Au (Gold) - Palik.txt",unpack='true')
lam=wavelength_nm
n=fdtd_n+1j*fdtd_k

epsilon1=fdtd_n**2-fdtd_k**2
epsilon2=2*fdtd_n*fdtd_k
epsilon = epsilon1+1j*epsilon2
epsilon_s=1.8

n_s=np.sqrt(epsilon_s)
k_0=2*np.pi/(lam*1e-9)
k_s=k_0*np.sqrt(epsilon_s)


def alpha_p(a,i):
    return a**3*(epsilon[i]-epsilon_s)/(epsilon[i]+2*epsilon_s)

####################################################################

def j(x):
    return np.sin(x)/x**2-np.cos(x)/x


def jj(x):
    return np.sin(x)+np.cos(x)/x-np.sin(x)/x**2        


def h(x):
    return np.exp(1j*x)*(-1/x-1j/x**2)        


def hh(x):
    return -1j*np.exp(1j*x)+(x+1j)*np.exp(1j*x)/x**2


def alpha_mie(a,i):
    k=k_s[i]
    nn=n[i]/n_s
    rho =k*a  
    a1=(nn**2*j(nn*rho)*jj(rho)-j(rho)*jj(nn*rho))/(nn**2*j(nn*rho)*hh(rho)-h(rho)*jj(nn*rho))
    return 1j*3*a1/k**3/2

ext=np.zeros((1000),dtype=np.float)
ext2=np.zeros((1000),dtype=np.float)
for i in np.arange(1000):
    ext[i]=4*np.pi*k_s[i]*(alpha_mie(5e-9,i)).imag
    ext2[i]=4*np.pi*k_s[i]*(alpha_p(5e-9,i)).imag
plt.figure()
plt.plot(lam,ext,label="$mie$")
plt.plot(lam,ext2,label="$drude$")
plt.xlabel("$\lambda$")
plt.ylabel("$\sigma_{ext}$")
plt.title("cd_deep:y-x")
plt.legend() 
