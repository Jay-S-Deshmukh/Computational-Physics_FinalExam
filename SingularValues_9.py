# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:55:38 2020

@author: Jay
"""

import numpy as np

def singular(M):
    
    S = np.dot(np.transpose(M),M)
    s = np.linalg.eigvals(S)
    
    return np.sqrt(s)

A = np.array([[2,1],[1,0],[0,1]])
B = np.array([[1,1,0],[1,0,1],[0,1,1]])

print("Singular values of A are",singular(A))
print("Singular values of B are",singular(B))
