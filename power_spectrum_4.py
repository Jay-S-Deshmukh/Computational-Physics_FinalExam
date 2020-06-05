# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:10:43 2020

@author: Dell
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1024
n_bins = 20 
X = np.random.rand(N)

X_k = np.fft.fft(X)
P = (1/N)*np.abs(X_k)**2
k = 2*np.pi*np.fft.fftfreq(N,1)

k_bins = 5
kmax = np.amax(k)
kmin = np.amin(k)
dk = (kmax - kmin)/(k_bins)

print("Maximum value of wavevector k is",kmax)
print("Maximum value of wavevector k is",kmin)

P_binned_vals = np.zeros((k_bins,2))
for i in range(k_bins):
    for ki in k:
        if ki >= (kmin + i*dk) and ki <= (kmin + (i+1)*dk):
            P_binned_vals[i][0] = P_binned_vals[i][0] + P[i]
            P_binned_vals[i][1] = P_binned_vals[i][1] + 1
            
P_binned = np.zeros(N)
for i in range(N):
    j = int(np.floor(5*i/N))
    P_binned[i] = P_binned_vals[j][0]/P_binned_vals[j][1]
    
plt.figure(figsize=(7,7))
n1, bins1, patches1 = plt.hist(X, n_bins, facecolor='blue', density='true', alpha=0.5, label='np.random.rand')
plt.axhline(y=1, xmin=0, xmax=1,label='Uniform')
plt.title("np.random.rand; n=1024")
plt.legend()
plt.show()

idx = np.argsort(k)
plt.plot(k[idx],P[idx],'b',label='Unbinned')
plt.plot(k[idx],P_binned[idx],'r',label='Binned')
plt.legend()
plt.title("Power Spectrum")
plt.show()

