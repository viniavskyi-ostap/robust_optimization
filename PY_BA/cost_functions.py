import numpy as np
from robust_kernels import *
from math_utils import *
import math

class NLSQ_Residuals:
    """
    Handle residuals and their respective weights and  Jacobians
    """
    def __init__(self, usedParamTypes, nMeasurements, measurementDimension, paramDesc):
        self.residuals = np.zeros((nMeasurements, measurementDimension), dtype=np.double)
        self.weights = np.zeros(nMeasurements, dtype=np.double)
        self.Js = []
        for k in range(len(usedParamTypes)):
            paramType = usedParamTypes[k]
            paramDimension = paramDesc.dimension[paramType]
            self.Js.append( [np.zeros(( measurementDimension, paramDimension), dtype=np.double ) ]*nMeasurements )


class BundleCostFunction:
    """
    Handle the Bundle Adjustment Function
    """
    def __init__(self, paramDesc, usedParamTypes, inlierThreshold, cams, Xs,
                 measurements, correspondingParams):
        
        self.measurementDimension = 2
        self.paramDesc = paramDesc
        self.usedParamTypes = usedParamTypes
        self.measurements = measurements
        self.correspondingParams = correspondingParams

        # Initialize the robust kernel
        self.psi = Psi_SmoothTrunc()
        self.alpha = 1.0
        self.tau2 = 1.0
        self.scale2 = 1.0
        # These two contain optimizable parameters
        self.cams = cams
        self.Xs = Xs    
        
        #
        self.nMeasurements = measurements.shape[0]

        #Derivatives
        self.dp_dRT = [np.zeros((2,6), dtype = np.double) for i in range(self.nMeasurements)]
        self.dp_dX =  [np.zeros((2, 3), dtype=np.double) for i in range(self.nMeasurements)]

        #Residuals        
        self.residuals = np.zeros((self.nMeasurements, self.measurementDimension), dtype=np.double)
        # Store the IRLS Weights
        self.irlsWeights = np.ones(self.nMeasurements, dtype=np.double)
        
        #Jacobian
        self.Js = []
        for k in range(len(usedParamTypes)):
            paramType = self.usedParamTypes[k]
            paramDimension = self.paramDesc.dimension[paramType]
            self.Js.append( [[np.zeros(( self.measurementDimension, paramDimension), dtype=np.double ) ] for i in range(self.nMeasurements)] )

        #Errors (before applying the robust kernel)
        self.errors = np.zeros(self.nMeasurements)
        

    def poseDerivatives(self, i, j):
        """
        This computes the transformed point XX = R*X + T
        and also computes the derivatives dXX_dRT and dXX_dX
        """
        d_dRT = np.zeros((3,6), dtype=np.double)        
        XX = self.cams[i].transformPointIntoCameraSpace(self.Xs[j])

        # See Frank Dellaerts bundle adjustment tutorial
        # d(dR * R0 * X + T)/d_omega = -[R0 * X]_x
        # d(dR * R0 * X + T)/d_T = Identity        
        J = -1.0 * makeCrossProductMatrix(XX - self.cams[i].T)
        d_dRT[:,3:6] = J
        d_dRT[:,0:3] = np.eye(3)

        d_dX = self.cams[i].R

        return XX, d_dRT, d_dX

       
    def precompute_residuals(self):        
        for k in range(self.nMeasurements):
            view = self.correspondingParams[k][0]
            point = self.correspondingParams[k][1]
            q = self.cams[view].projectPoint(self.Xs[point, :])
            self.residuals[k] = q - self.measurements[k]
            self.errors[k] = (np.linalg.norm(self.residuals[k]))**2


        self.robustCost = 0    
        for k in range(self.nMeasurements):            
            cost = self.psi.fun(self.errors[k])                        
            self.robustCost += cost
            # if k%1000 ==0: #DEBUG
            #     print(k, self.errors[k], cost, self.robustCost)

            
    
    def precompute_bundle_derivatives(self):
        for k in range(self.nMeasurements):
            view = self.correspondingParams[k][0]
            point = self.correspondingParams[k][1]

            XX, dXX_dRT, dXX_dX = self.poseDerivatives(view, point)

            # Now, compute the projected image point
            xu = np.zeros(2)
            xu[0] = XX[0]/XX[2]
            xu[1] = XX[1]/XX[2]

            #xd = self.cams[view].distortion.apply(xu)
            focalLength = self.cams[view].getFocalLength()

            dp_dxd = np.zeros((2,2), dtype=np.float64)
            dp_dxd[0][0] = focalLength
            dp_dxd[0][1] = 0

            dp_dxd[1][0] = 0
            dp_dxd[1][1] = focalLength

            dxu_dXX = np.zeros((2,3) , dtype=np.double)
            dxu_dXX[0][0] = 1.0/XX[2]   
            dxu_dXX[0][1] = 0.0
            dxu_dXX[0][2] = -XX[0]/(XX[2]*XX[2])
            
            dxu_dXX[1][0] = 0
            dxu_dXX[1][1] = 1.0/XX[2]
            dxu_dXX[1][2] = -XX[1]/(XX[2]*XX[2])


            dxd_dxu = self.cams[view].distortion.derivativeWrtUndistortedPoint(xu)
            dp_dxu = dp_dxd.dot(dxd_dxu)

            dp_dXX = dp_dxu.dot(dxu_dXX)
            
            self.dp_dRT[k] = dp_dXX.dot(dXX_dRT)
            self.dp_dX[k]  = dp_dXX.dot(dXX_dX)
            
    def camera_Jacobian(self,k):
        return self.dp_dRT[k]
                        
    # Implementaions for the optimization process  
    def preIterationCallback(self):
        self.precompute_residuals()
        self.precompute_bundle_derivatives()

    def initializeResiduals(self):
        self.precompute_residuals()

    def evalResidual(self, k):
        return self.residuals[k]

    def cache_IRLS_Weights(self):
        for k in range(self.nMeasurements):
            tau_scaled = self.tau2 * self.scale2
            w = self.psi.weight_fun(self.errors[k]/tau_scaled)
            self.irlsWeights[k] = math.sqrt(w)            

    def clip_IRLS_Weights(self, M):
        max_weight = max(self.irlsWeights)
        min_weight = min(self.irlsWeights)
        print("Max - Min IRLS Weights = ", max_weight, min_weight)
        for k in range(self.nMeasurements):            
            self.irlsWeights[k] = max(self.irlsWeights[k], M*max_weight)

                
    def scaleResidualsByWeights(self):
        for k in range(self.nMeasurements):
            self.residuals[k,:] *= self.irlsWeights[k]            

    def fillJacobian(self, whichParam, k):
        if whichParam == 0: # CAMERA:
            return self.camera_Jacobian(k)
        elif whichParam == 1:
            return self.dp_dX[k]
        else:
            assert(False)
    

    def fillAllJacobians(self):        
        for i in range(len(self.usedParamTypes)):
            J = self.Js[i]
            for k in range(self.nMeasurements):
                J[k] = self.fillJacobian(i, k)
                J[k] = self.irlsWeights[k]*J[k]

    
    
    




        
            
