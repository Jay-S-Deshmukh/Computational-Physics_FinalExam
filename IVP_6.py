# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:36:33 2020

@author: Jay
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

a=0
b=0.5
N=51
h=(b-a)/(N-1)

def func(Y,x):
    
    v1 = 32*Y[0] + 66*Y[1] + 2*x/3 + 2/3
    v2 = -66*Y[0] -133*Y[1] - 1*x/3 - 1/3
    
    V = np.array([v1,v2])
    return V

x_list = np.linspace(a,b,N)
Y = np.zeros((N,2))


Y[0]=[1/3,1/3]

for i in range(N-1):

    k1 = h*func(Y[i],x_list[i])
    k2 = h*func(Y[i]+k1/2,x_list[i]+h/2) 
    k3 = h*func(Y[i]+k2/2,x_list[i]+h/2)
    k4 = h*func(Y[i]+k3,x_list[i]+h)
    Y[i+1] = Y[i] + (k1+2*k2+2*k3+k4)/6
 
ode_res = odeint(func,[1/3,1/3],x_list)

plt.plot(x_list,Y[:,0],'r',label='RK4 y1')
plt.plot(x_list,Y[:,1],'b',label='RK4 y2')
plt.plot(x_list,ode_res[:,0],'g^',label='Odeint y1')
plt.plot(x_list,ode_res[:,1],'y^',label='Odeint y2')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Coupled IVP")
plt.legend()
plt.show()

plt.plot(x_list,Y[:,0]+2*Y[:,1],'r',label='y1 + 2y2')
plt.plot(x_list,np.exp(-100*x_list),'b^',label='e^(-100x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Verification")
plt.legend()
plt.show()