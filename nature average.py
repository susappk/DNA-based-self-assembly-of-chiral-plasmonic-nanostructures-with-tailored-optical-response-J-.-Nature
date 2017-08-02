#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 10:40:12 2017

@author: xiaabs
"""

import numpy as np
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
import statsmodels.api as sm
import peak_deep as pd

wavelength_nm,exp_n,exp_k,fdtd_n,fdtd_k=np.loadtxt("/home/xiaabs/Desktop/Au (Gold) - Johnson and Christy.txt",unpack='true')
Sample=301
lam=wavelength_nm
n=exp_n+1j*exp_k


epsilon1=fdtd_n**2-fdtd_k**2
epsilon2=2*fdtd_n*fdtd_k
epsilon = epsilon1+1j*epsilon2
epsilon_s=1.8

k_0=2*np.pi/(lam*1e-9)
n_s=np.sqrt(epsilon_s)
k_s=k_0*n_s


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

''' propagater use k_s '''
def G(r,i): 
    r=r.reshape(1,3)  
    R=np.linalg.norm(r)        
    I=np.eye(3,dtype=np.complex)
    nn=r.T.dot(r)/R**2          
    squre=((1+1j/k_s[i]/R-1/k_s[i]**2/R**2)*I+(-1-1j*3/k_s[i]/R+3/k_s[i]**2/R**2)*nn)       
    propagater=squre*np.exp(1j*k_s[i]*R)/(4*np.pi*R)
    return  propagater

class main():
    def __init__(self,edge,period,radius,number):
        self.radius=1e-9*radius  
        self.number=number
        self.wei=self.position(edge,period)


    def position(self,edge,period):
        AA=5
        e=period/AA*1e-9  
        edge=edge*1e-9  
        out={}
        for i in np.arange(self.number):
            theta=i*(2*np.pi/AA)
            out[i]= np.array([np.cos(theta)*edge,np.sin(theta)*edge,-e*i])
        return out
    

    def ext(self,i):
        I=np.eye(3*self.number,dtype=np.complex)        
        A=np.zeros((3*self.number,3*self.number),dtype=np.complex)
        for a in np.arange(self.number):
            for b in np.arange(self.number): 
                if not(a==b):                                           
                    A[(0+3*a):(3+3*a),(0+3*b):(3+3*b)]=4*np.pi*k_s[i]**2*G(self.wei[a]-self.wei[b], i) 
        A=np.linalg.inv(I/alpha_mie(self.radius,i)-A)
        
        def imag_a(A):
            return 1/(2j)*(A-A.T.conj())
            
            
        def a_ij(A):
            return np.array([imag_a(A[2,1])-imag_a(A[1,2]),
                             imag_a(A[0,2])-imag_a(A[2,0]),
                             imag_a(A[1,0])-imag_a(A[0,1])])
        
            
        def j1(x):
            return np.sin(x)/x**2-np.cos(x)/x
            
            
        out=0
        for a in np.arange(self.number):
            for b in np.arange(self.number):
                if not(a==b):   
                    r=self.wei[a]-self.wei[b]
                    r_mo=np.linalg.norm(r)
                    r_norm=r/r_mo
                    out += 4*np.pi*k_s[i]*j1(k_s[i]*r_mo)*a_ij(A[(0+3*a):(3+3*a),(0+3*b):(3+3*b)]).dot(r_norm)
        return out   

        
def run(edge,period):
    if __name__=='__main__':
        print('Parent process %s.' % os.getpid())
        p = Pool(8)
        result={}
        for i in range(Sample):
            result[i]=p.apply_async(main(edge=edge,period=period,radius=5,number=6).ext, args=(i,))
        p.close()    
        p.join()  
        
        
    return result
#
#""" four particle best period """
#pp=np.sqrt(27/8)*25
#result=run(edge=25,period=pp)
#cd=np.zeros((Sample),dtype=np.float)
#for i in np.arange(Sample):
#    cd[i]=result[i].get()
#plt.figure()   
#plt.plot(lam,cd,label="$y=x$")
#plt.xlabel("$\lambda$")
#plt.ylabel("$\sigma_{ext}$")
#plt.title("cd_deep:y-x"+str(pp))
#plt.legend()  



#R=25
#n=10
#per=np.sqrt(1/(1-1/n**2))*2*R*np.sin(np.pi/n)
#print(per)
""" five particle best period """
#
#pp=np.sqrt(32/15)*25
#result=run(edge=25,period=pp)
#cd=np.zeros((Sample),dtype=np.float)
#for i in np.arange(Sample):
#    cd[i]=result[i].get()
#plt.figure()   
#plt.plot(lam,cd,label="$y=x$")
#plt.xlabel("$\lambda$")
#plt.ylabel("$\sigma_{ext}$")
#plt.title("cd_deep:y-x"+str(pp))
#plt.legend()  

#
ra=1
pp=30
result=run(edge=25*ra,period=pp)
cd=np.zeros((Sample),dtype=np.float)
for i in np.arange(Sample):
    cd[i]=result[i].get()
plt.figure()   
plt.plot(lam,cd,label="$y=x$")
plt.xlabel("$\lambda$")
plt.ylabel("$\sigma_{ext}$")
plt.title("cd_deep:y-x"+str(pp))
plt.legend()
 
#mode = "z_direction"
#if mode == "edge":
#    Edge=np.linspace(10,50,10)
#    
#    
#elif mode == "z_direction":
#    sam=15
#    Edge=np.linspace(30,90,sam)
#    
#    
#
#ext={"cd":np.zeros((Sample,sam),dtype=np.float),"deep":np.zeros((sam),dtype=np.float),"peak":np.zeros((sam),dtype=np.float)}
#point=np.zeros((sam),dtype=np.float)
#
#
#for m in np.arange(sam):
#    if mode == "edge":
#        
#        result=run(edge=Edge[m],period=57)
#    elif mode == "z_direction":
#        result=run(edge=17,period=Edge[m])
#    for i in np.arange(Sample):
#        ext["cd"][i,m]=result[i].get()
#    #ext["deep"][m]=np.min(ext["cd"][:,m])
#    #ext["peak"][m]=np.max(ext["cd"][:,m])
#    
#    ext["deep"][m]=pd.wavedeep(ext["cd"][:,m],lam,mode = "number")[0]
#    ext["peak"][m]=pd.wavepeak(ext["cd"][:,m],lam,mode = "number" )[0]
#
#    print(pd.wavedeep(ext["cd"][:,m],lam,mode = "str"))
#    print(pd.wavepeak(ext["cd"][:,m],lam,mode = "str"))
#    plt.figure()   
#    plt.plot(lam,ext["cd"][:,m],label="$y=x$")
#    plt.xlabel("$\lambda$")
#    plt.ylabel("$\sigma_{ext}$")
#    plt.title("cd_deep:y-x"+str(Edge[m]))
#    plt.legend()  
#
#plt.figure()
#plt.plot(Edge,ext["deep"],label="$y=x$")
#plt.xlabel("edge")
#plt.ylabel("$\sigma_{ext}$")
#plt.title("cd_deep:y-x"+str(Edge[m]))
#plt.legend() 
#
#
##y=np.log(-ext["deep"])
##x=Edge
##X=sm.add_constant(x)
##regression= sm.OLS(y,X).fit()
###print(regression.summary())
##y_fitted=regression.fittedvalues
##plt.figure()
##plt.plot(x,y,"o",label="$data$")
##plt.plot(x,y_fitted,label="$fitted$")
##plt.xlabel(mode)
##plt.ylabel("$\sigma_{ext}$")
##plt.title("$e^x$"+str(regression.params))
##plt.legend()
##plt.figure()
#
##
##y=np.log(-ext["deep"])
##x=np.log(Edge)
##X=sm.add_constant(x)
##regression= sm.OLS(y,X).fit()
###    print(regression.summary())
##y_fitted=regression.fittedvalues
##plt.figure()
##plt.plot(x,y,"o",label="$data$")
##plt.plot(x,y_fitted,label="$fitted$")
##plt.xlabel(mode)
##plt.ylabel("$\sigma_{ext}$")
##plt.title("$x^n$ deep"+str(regression.params))
##plt.legend()
##plt.figure()
##
##y=np.log(ext["peak"])
##x=np.log(Edge)
##X=sm.add_constant(x)  
##regression= sm.OLS(y,X).fit()
###    print(regression.summary())
##y_fitted=regression.fittedvalues
##plt.figure()
##plt.plot(x,y,"o",label="$data$")
##plt.plot(x,y_fitted,label="$fitted$")
##plt.xlabel(mode)
##plt.ylabel("$\sigma_{ext}$")
##plt.title("$x^n peak$"+str(regression.params))
##plt.legend()
##plt.figure()
##
