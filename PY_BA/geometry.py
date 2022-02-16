# This file takes care of the necessary components for 3D geoemtry
from __future__ import division
import numpy as np
import math
#from math_utils import *
from scipy.spatial.transform import Rotation

def createRotationMatrixRodrigues(omega):
    """
    create Rotation Matrix from omega vector
    """
    R =  Rotation.from_rotvec(omega).as_matrix()
    return R

#------------------------------------------------------
def createRodriguesParamFromRotationMatrix(R):
    """
    Convert from a Rotation Matrix to a vector
    """
    omega = Rotation.from_matrix(R).as_rotvec()
    return omega

#------------------------------------------------------
class Camera:
    """
    Camera Matrices
    """
    def __init__(self):
        self.K = np.eye(3)
        self.R = np.eye(3)
        self.T = np.zeros(3)
        self.q = np.zeros(3)

        #Each camera should have a distortion function
        self.distortion = SimpleDistortionFunction()

    def setIntrinsicMatrixFromF(self, f):
        self.K[0][0] = -1.0*f
        self.K[1][1] = -1.0*f

    def setTranslation(self, T):
        self.T = T

    def setRotationMatrix(self, R):
        self.R = R

    def setRotationFromOmega(self, omega):
        self.R = createRotationMatrixRodrigues(omega)

    def setRotationVector(self, q):
        self.q = q

    def getRotationMatrix(self):
        return self.R

    def getTranslation(self):
        return self.T

    def getFocalLength(self):
        return self.K[0][0]
        
    def transformPointIntoCameraSpace(self, XX):
        Xp = self.R.dot(XX) + self.T
        return Xp

    def projectPoint(self, X):
        XX = self.transformPointIntoCameraSpace(X)
        xu = np.array([XX[0]/XX[2], XX[1]/XX[2]])
        xd = self.distortion.apply(xu)        
        return self.getFocalLength()*xd
                      
    
#*************************************************************
class SimpleDistortionFunction:
    """
    Camera Distortion function
    """
    def __init__(self):
        self.k1 = 0
        self.k2 = 0

    def setKParams(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def apply(self, xu):
        r2 = xu[0] * xu[0] + xu[1] * xu[1]
        r4 = r2 * r2
        kr = 1 + self.k1*r2 + self.k2*r4

        xd = np.array([kr * xu[0], kr*xu[1]])
        return xd
        
    def derivativeWrtUndistortedPoint(self, xu):
        r2 = xu[0] * xu[0] + xu[1] * xu[1]
        r4 = r2 * r2
        kr  = 1 + self.k1*r2 + self.k2*r4
        dkr = 2 * self.k1 + 4 * self.k2 * r2

        deriv = np.zeros((2,2))
        deriv[0][0] = kr + xu[0] * xu[0] * dkr
        deriv[0][1] = xu[0] * xu[1] * dkr
        deriv[1][0] = deriv[0][1]
        deriv[1][1] = kr + xu[1] * xu[1] * dkr

        return deriv

    

    
    
   # def 
