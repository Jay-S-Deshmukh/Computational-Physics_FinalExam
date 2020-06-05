# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:27:43 2020

@author: Jay
"""

import numpy as np
from matplotlib import pyplot as plt

def f(x):
    if(x>-1 and x<1):
        return 1
    return 0

def f_ft(k):
    if(k==0):
        return np.sqrt(2/np.pi)
    else: return np.sqrt(2/np.pi)*np.sin(k)/k

xmin = -7.0
xmax = 7.0
n_points = 128
dx = (xmax - xmin)/(n_points-1)

x = np.linspace(xmin,xmax,n_points)
sampled_data = [f(xi) for xi in x] 
  
nft = np.fft.fft(sampled_data, norm='ortho')
k = np.fft.fftfreq(n_points,dx)
k = 2*np.pi*k
factor = np.exp(-1j*k*xmin)

aft = dx*np.sqrt(n_points/(2*np.pi))*factor*nft

aft = np.fft.fftshift(aft)
k = np.fft.fftshift(k)


ft_analytical = [f_ft(ki) for ki in k]
box = [f(xi) for xi in x]

#plt.plot(x,box,'g',label='Box Function')
plt.plot(k,np.real(aft),'r^',label='Numerical')
plt.plot(k,ft_analytical,'b',label='Analytical')
plt.xlabel('k')
plt.ylabel('f(k)')
plt.legend()
plt.title("FT of Box function")
plt.show()

plt.plot(x,box,'g',label='Box Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title("Box function")
plt.show()
