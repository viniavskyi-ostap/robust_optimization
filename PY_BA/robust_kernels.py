# This file stores the commonly used robust kernels
import numpy as np
import math

class Psi_SmoothTrunc():
    def fun(self, r2):
        cost = 0.25*(r2*(2.0 - r2) if r2 <= 1.0 else 1.0)        
        return cost
    def weight_fun(self, r2):
        return np.maximum(0.0, 1.0 - r2)

class Psi_LeastSquares():
    def fun(self, r2):
        cost = math.sqrt(r2) 
        return cost
    def weight_fun(self, r2):
        return 1.0/math.sqrt(r2) 


    
    
