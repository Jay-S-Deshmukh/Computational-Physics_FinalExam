# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:52:15 2020

@author: Jay
"""
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

e = np.exp(1)
err=0.001

a = 0
b = 1
h = 0.05
N = int((b-a)/h) + 1

def f_analytical(t):
    return (e**2/(e**4-1))*(e**(2*t) - e**(-2*t)) + t

def f(Y,x):
    
    return np.array([Y[1],4*(Y[0]-x)])

def ivp_RK4(func,Y0,x_list):
    
    Y = np.zeros((N,2))
    Y[0]=Y0
    
    for i in range(N-1):
    
        k1 = h*func(Y[i],x_list[i])
        k2 = h*func(Y[i]+k1/2,x_list[i]+h/2) 
        k3 = h*func(Y[i]+k2/2,x_list[i]+h/2)
        k4 = h*func(Y[i]+k3,x_list[i]+h)
        Y[i+1] = Y[i] + (k1+2*k2+2*k3+k4)/6
    
    return Y
    
def f_0(x,Y):
    
    Z = f(Y,x)
    return Z

def bc_0(ya,yb):
    
    return np.array([ya[0] - 0,yb[0] - 2.0])

x_list = np.linspace(a,b,N)
sol = np.zeros((N,2))

x_f=2
v=100
sol = ivp_RK4(f, [0,v], x_list)
while(sol[-1,0]>(x_f+err) or sol[-1,0]<(x_f-err)):
    
    sol_dv = (ivp_RK4(f, [0,v+h], x_list)[-1,0] - sol[-1,0])/h
    v = v - (sol[-1,0] - x_f)/sol_dv
    sol = ivp_RK4(f, [0,v], x_list)

sol_analytical = f_analytical(x_list)

err_pc = np.zeros(N)
for i in range(N):
    
    if i!=0:
        err_pc[i] = 100*np.abs((sol_analytical[i] - sol[i,0])/(sol_analytical[i]))
    print("Relative error at x =", np.round(x_list[i],2), "is", err_pc[i],"%")

Y_0 = np.zeros((2, x_list.size))
sol_0 = solve_bvp(f_0, bc_0, x_list, Y_0)

plt.plot(x_list,err_pc,'r',label='Percentage Error')
plt.title('Shooting method')
plt.xlabel('x')
plt.ylabel('Error %')
plt.legend()
plt.show()
    
plt.plot(x_list,sol[:,0],'r',label='Shooting method')
plt.plot(sol_0.x,sol_0.y[0],'b^',label='solve_bvp',markersize=7)
plt.plot(x_list,sol_analytical,'y.',label='Analytical sol')
plt.title('BVP')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()