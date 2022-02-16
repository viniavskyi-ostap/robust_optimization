"""
Classes required for non-linear least squares
This file particularly
"""
from __future__ import division
from geometry import *
from math_utils import *
from robust_kernels import Psi_SmoothTrunc
from cost_functions import BundleCostFunction, NLSQ_Residuals
from scipy.sparse import csc_matrix, csr_matrix, bsr_matrix, linalg as sla
from scipy.sparse.linalg import lsqr, spsolve as scipy_spsolve
# from scikits.umfpack import spsolve
from scipy.linalg import solve
from sksparse.cholmod import cholesky, CholmodError

from numba import jit, float32

NLSQ_MAX_PARAM_TYPES = 3
#@jit(nopython=True)
class NLSQ_Optimizer:
    """
    This class contains main procedures for non-linear least squares optimization
    """
    def __init__(self, paramDesc, costFunctions, \
                 clipIRLS_Weights = False, log_file = 'running_logs.txt'):
        self.paramDesc = paramDesc
        self.costFunctions = costFunctions        
        
        self.cams = costFunctions[0].cams
        self.Xs = costFunctions[0].Xs

        self.paramTypeStartId =  [0 for i in range(NLSQ_MAX_PARAM_TYPES)]
        self.paramTypeRowStart = [0 for i in range(NLSQ_MAX_PARAM_TYPES)]

        self.hessianIndices = []
        self.hessian = BlockHessian()
        self.residuals = []
        self.Js = []

        # Intitialize some other variables
        for paramType in range(paramDesc.nParamTypes): # Block params
            self.paramTypeStartId[paramType+1] = self.paramTypeStartId[paramType] + paramDesc.count[paramType]
        
        self.totalParamCount = self.paramTypeStartId[paramDesc.nParamTypes]
        self.paramInverseMap = [[(0,0)] for i in range(self.totalParamCount)]

        for paramType in range(paramDesc.nParamTypes):
            for ix in range(paramDesc.count[paramType]):
                id = self.getParamId(paramType, ix)
                self.paramInverseMap[id] = (paramType, ix)

        self.setupSparseJtJ()

        self.JtJ_size = 0
        for t in range(self.paramDesc.nParamTypes):
            self.paramTypeRowStart[t] = self.JtJ_size
            self.JtJ_size += self.paramDesc.dimension[t]*self.paramDesc.count[t]
        self.paramTypeRowStart[self.paramDesc.nParamTypes] = self.JtJ_size

        self.totalParamDimension = self.JtJ_size
        #self.sparseHessian = coo_matrix((self.totalParamDimension, self.totalParamDimension), dtype='double')

        # Other parameters
        self.currentIteration = 0
        self.maxIterations = 100
        self.minIterations = 10
        self.tau = 1e-3
        self.lamda = 1e-3
        self.gradientThreshold = 1e-3
        self.updateThreshold = 1e-8
        self.improvementThreshold = 1e-8
        self.nu = 2.0
        self.min_damping_value = 1e-5
        self.max_damping_value = 1e8
        
        self.damping_value = 1e-3

        # Storage for sparse Hessian matrix
        self.col_idxs = []
        self.row_idxs = []
        self.JtJ_data = []

        #Clip IRLS
        self.clipIRLS_Weights = clipIRLS_Weights
        self.irls_init_M = 0.02
        self.irls_clip_M = 0.02
        self.decrease_threshold = 50
        self.M_increse_rate = 0.5

        #logs
        self.logs = Logger(outputFile=log_file)



    def saveParamsToVector(self):
        """
        Save camera and point parameters to a vector
        """
        pos = 0
        x_saved = np.zeros(self.totalParamDimension)
        for i in range(len(self.cams)):
            x_saved[pos:pos+3] = self.cams[i].T
            pos+=3
        for i in range(len(self.Xs)):
            x_saved[pos:pos+3] = self.Xs[i,:]
            pos+=3
        for i in range(len(self.cams)):
            x_saved[pos:pos+3] = createRodriguesParamFromRotationMatrix(self.cams[i].R)
            pos+=3
        
        return x_saved

    def loadParamsFromVector(self, x_saved):
        """
        Load saved parameters from a vector to the cameras and points
        """
        pos = 0 
        for i in range(len(self.cams)):
            self.cams[i].T = x_saved[pos:pos+3]
            pos+=3
        
        for i in range(len(self.Xs)):
            self.Xs[i,:] = x_saved[pos:pos+3]
            pos+=3
        
        for i in range(len(self.cams)):
            self.cams[i].R = createRotationMatrixRodrigues(x_saved[pos:pos+3])
            pos+=3
        

    def getParamId(self, paramType, paramIx):
        return self.paramTypeStartId[paramType] + paramIx

    def setupSparseJtJ(self):
        """
        Hessian blocks are stored in a consecutive array. This array is managed by self.hessian, which is 
        a BlockHessian Object. For the k-th measurement, self.hessianIndices[k][variable1][variable2] will 
        point to the particular block of Hessian.        
        """
        nObjs = len(self.costFunctions)
        nonZeroPosMaps = [[dict() for i in range(NLSQ_MAX_PARAM_TYPES)] for j in range(NLSQ_MAX_PARAM_TYPES)]
        nNonZeroBlocks = 0

        for obj in range(nObjs):
            costFun = self.costFunctions[obj]
            nParamTypes = len(costFun.usedParamTypes)
            nMeasurements = costFun.nMeasurements
            
            #Prepare Hessian
            self.hessianIndices.append( [[ [ 0 for i in range(nParamTypes) ] for j in range(nParamTypes) ] for k in range(nMeasurements)])

            #Prepare Residuals and Jacobian
            self.residuals.append(NLSQ_Residuals(costFun.usedParamTypes, nMeasurements, \
                                 costFun.measurementDimension, self.paramDesc ))
                        
            for k in range(nMeasurements):
                for i1 in range(nParamTypes):
                    t1 = costFun.usedParamTypes[i1]
                    ix1 = costFun.correspondingParams[k][t1]                                        

                    for i2 in range(nParamTypes):
                        t2 = costFun.usedParamTypes[i2]
                        ix2 = costFun.correspondingParams[k][t2]                        
                        nzPosMap = nonZeroPosMaps[t1][t2]

                        if ((ix1, ix2) in nzPosMap.keys())==False:
                            curPos = len(nzPosMap)
                            self.hessianIndices[obj][k][i1][i2] = curPos
                            nzPosMap.update({(ix1, ix2): curPos})
                            self.hessian.nonZeroPairs[t1][t2].append((ix1,ix2))
                            nNonZeroBlocks+=1
                        else:
                            self.hessianIndices[obj][k][i1][i2] = nzPosMap[(ix1, ix2)]

            self.hessian.allocateMatrix(self.paramDesc)



    def fillJacobians(self):
        """
        Trigger the cost functions to fill the Jacobians. The Jacobians are stored in costFun.Js
        """
        nObjs = len(self.costFunctions)
        for obj in range(nObjs):            
            costFun = self.costFunctions[obj]            
            costFun.precompute_bundle_derivatives()
            costFun.fillAllJacobians()


    def fillHessian(self):
        """
        Fill Hessian approximation (J'J + lambda I) to the Hessian Blocks (stored in self.Hessian)
        """
        self.hessian.setZero()
        nObj = len(self.costFunctions)
        for obj in range(nObj):
            costFun = self.costFunctions[obj]
            nParamTypes = len(costFun.usedParamTypes)
            nMeasurements = costFun.nMeasurements
                        
            #residuals =  costFun.residuals

            for i1 in range(nParamTypes):
                t1 = costFun.usedParamTypes[i1]
                #dim1 = self.paramDesc.dimension[t1]
                Js1 = costFun.Js[i1]

                for i2 in range(nParamTypes):
                    t2 = costFun.usedParamTypes[i2]
                    #dim2 = self.paramDesc.dimension[t2]
                    Js2 = costFun.Js[i2]

                    if len(self.hessian.Hs[t1][t2])==0:
                        continue
                    
                    Hs = self.hessian.Hs[t1][t2]                    
                    for k in range(nMeasurements):
                        #ix1 = costFun.correspondingParams[k][i1]
                        #id1 = self.getParamId(t1, ix1)                        
                        #ix2 = costFun.correspondingParams[k][i2]
                        #id2 = self.getParamId(t2, ix2)                        
                        n = self.hessianIndices[obj][k][i1][i2]
                        JtJ = Js1[k].transpose().dot(Js2[k])
                        Hs[n] += JtJ


    def evalJt_e(self):
        """
        After filling all the jacobians, the gradient vector is computed by J'e
        """
        Jt_e = np.zeros(self.totalParamDimension, dtype=np.float64)
        nObjs = len(self.costFunctions)
                   
        for obj in range(nObjs):
            costFun = self.costFunctions[obj]
            residuals = costFun.residuals
            nParamTypes = len(costFun.usedParamTypes)
            nMeasurements = costFun.nMeasurements                

            for i in range(nParamTypes):
                paramType = costFun.usedParamTypes[i]
                paramDim = self.paramDesc.dimension[paramType]
                J = costFun.Js[i]

                for k in range(nMeasurements):
                    id = costFun.correspondingParams[k][i]
                    dstRow = self.paramTypeRowStart[paramType] + id*paramDim                                        
                    Jkt_e = J[k].transpose().dot(residuals[k].reshape((2,1)))       
                    for l in range(paramDim):
                        Jt_e[dstRow+l] += Jkt_e[l]

            return Jt_e

                
    def fillJtJ(self):
        """
        Fill Hessian to the sparse matrix
        """        
        self.row_idxs = []
        self.col_idxs = []
        self.JtJ_data = []        
        
        for t1 in range(self.paramDesc.nParamTypes):
            for t2 in range(self.paramDesc.nParamTypes):               
                nz = self.hessian.nonZeroPairs[t1][t2]
                Hs = self.hessian.Hs[t1][t2]
                if len(nz) == 0:
                    continue
                
                for k in range(len(nz)):                    
                    ix1 = nz[k][0]
                    ix2 = nz[k][1]
                    dim1 = self.paramDesc.dimension[t1]
                    dim2 = self.paramDesc.dimension[t2]
                    r0 = self.paramTypeRowStart[t1] + ix1 * dim1
                    c0 = self.paramTypeRowStart[t2] + ix2 * dim2

                    for r in range(dim1):
                        for c in range(dim2):                            
                            if (r0 + r >= c0 +c):
                                self.row_idxs.append(r0+r)
                                self.col_idxs.append(c0+c) 
                                self.JtJ_data.append(Hs[k][r][c])           
        
        self.sparseHessian = csc_matrix( (self.JtJ_data, (self.row_idxs, self.col_idxs) ), \
                             shape=(self.totalParamDimension, self.totalParamDimension), dtype=np.float64 )                        

                            

    def updateParameters(self, paramType, delta):
        """
        From the vector array delta, update all the parameters (cameras or points depending on the paramType)
        """
        if (paramType==0):
            for i in range(len(self.cams)):
                #Update Translation
                self.cams[i].T[0] += delta[i][0]
                self.cams[i].T[1] += delta[i][1]
                self.cams[i].T[2] += delta[i][2]

                #Update Rotation
                omega = delta[i][3:6]
                dR = createRotationMatrixRodrigues(omega)
                R1 = dR.dot(self.cams[i].R)
                self.cams[i].R = project_to_SO3(R1)

        elif (paramType==1):
            #Update 3D Points
            for i in range(self.Xs.shape[0]):
                self.Xs[i][0] += delta[i][0]
                self.Xs[i][1] += delta[i][1]
                self.Xs[i][2] += delta[i][2]


    def eval_current_cost(self):
        cost = 0
        for obj in range(len(self.costFunctions)):            
            self.costFunctions[obj].precompute_residuals()                
            cost += self.costFunctions[obj].robustCost
        return cost
    
    def compute_IRLS_Weights(self):
        for obj in range(len(self.costFunctions)):            
            self.costFunctions[obj].cache_IRLS_Weights()
            if (self.clipIRLS_Weights):
                self.costFunctions[obj].clip_IRLS_Weights(self.irls_clip_M)
                
            
    def scale_residuals_by_weights(self):
        for obj in range(len(self.costFunctions)):            
            self.costFunctions[obj].scaleResidualsByWeights()


    def minimize(self):
        """
        Main routine for the minimization process
        """

        nObjs = len(self.costFunctions)                 
        for currentIteration in range(self.maxIterations):                                    
            
            # First, compute cost                        
            initial_cost = self.eval_current_cost()

            #Also, compute IRLS Weights
            self.compute_IRLS_Weights()

            #Then, scale the residuals by the IRLS Weights
            self.scale_residuals_by_weights()
            print('----------------------------------------------------------------------')
            print("Current Iteration {}, Initial Cost = {}, Damping Value = {}".format(currentIteration, initial_cost, self.damping_value))                        
            self.logs.log(initial_cost)

            #Now, fill the Jacobians
            self.fillJacobians()                
            
            #Then, eval gradient by Jt_e
            Jt_e = self.evalJt_e()
            Jt_e = -1.0 * Jt_e
            
            #Fill Hessian information to the contiguous blocks
            self.fillHessian()            
            
            success_LDL = False
        
            # Augment Hessian
            for paramType in range(self.paramDesc.nParamTypes):
                Hs = self.hessian.Hs[paramType][paramType]
                nzPairs = self.hessian.nonZeroPairs[paramType][paramType]
                dim = self.paramDesc.dimension[paramType]
                count = len(nzPairs)

                for n in range(count):
                    if (nzPairs[n][0] == nzPairs[n][1]):
                        for l in range(dim):
                            Hs[n][l][l] += self.damping_value

            # Fill to the sparse matrix
            self.fillJtJ()

            # Now, solve J'J delta = -Jt_e
            print("Solving JTJ using Cholesky Factorization....")                        
            try:
                factor = cholesky(self.sparseHessian, ordering_method='colamd', mode='auto')
                delta = factor(Jt_e)                      
                success_LDL = True
            except CholmodError:
                print("Cholmod Error")

            success_decrease = False
            diff_cost = 0
            x_saved = None

            if (success_LDL):                                           
                #Before updating parameters, save them first just in case we need to restore:
                x_saved = self.saveParamsToVector()

                #Update the parameters 
                for paramType in range(self.paramDesc.nParamTypes):
                    paramDim = self.paramDesc.dimension[paramType]
                    count = self.paramDesc.count[paramType]
                    rowStart = self.paramTypeRowStart[paramType]
                    vectorArray = getVectorArray(count, paramDim, delta, rowStart)                
                    self.updateParameters(paramType, vectorArray)
                                    
                #Re-evaluate the objective
                current_cost = self.eval_current_cost()
                print("LDL Succeeded - Updating Parameters new cost = {}".format(current_cost))
                success_decrease = current_cost < initial_cost
                diff_cost = initial_cost - current_cost                

                self.irls_clip_M *= self.M_increse_rate

            else:                
                self.damping_value = min(self.max_damping_value, self.damping_value*10)
                print("LDL Failed - Increasing damping value to {}".format(self.damping_value))

            if (success_decrease):                                                    
                self.damping_value = max(self.min_damping_value, self.damping_value*0.1)
                if self.clipIRLS_Weights:
                    if diff_cost < self.decrease_threshold:
                        self.irls_clip_M = self.irls_init_M                                                                
                print("Update leads to cost reduction. Decreased Damping to = {}".format(self.damping_value))                

            else:
                if success_LDL:
                    self.loadParamsFromVector(x_saved)
                self.damping_value = min(1e8, self.damping_value*10)
                print("Update does not lead to cost reduction. Increased Damping to = {}".format(self.damping_value))
            

            
            
            
class NLSQ_ParamDesc:
    """
    Describe the parameter for the optimizer 
    """
    def __init__(self, nParamTypes):        
        self.nParamTypes = nParamTypes    # Number of parameters
        self.dimension = [0]*nParamTypes  # Dimensionality of each param type
        self.count = [0]*nParamTypes      # Number of variables of this type


class BlockHessian:
    """
    Definition to store and handle the the block Hessian
    """
    def __init__(self):
        self.Hs = [[ [] for i in range(NLSQ_MAX_PARAM_TYPES)] for j in range(NLSQ_MAX_PARAM_TYPES)]
        self.nonZeroPairs = [[ [] for i in range(NLSQ_MAX_PARAM_TYPES)] for j in range(NLSQ_MAX_PARAM_TYPES)]

    def allocateMatrix(self, paramDesc):
        nParamTypes = paramDesc.nParamTypes
        for t1 in range(nParamTypes):
            for t2 in range(nParamTypes):
                nz = len(self.nonZeroPairs[t1][t2])
                if nz > 0:
                    rows = paramDesc.dimension[t1]
                    cols = paramDesc.dimension[t2]
                    self.Hs[t1][t2] = [np.zeros((rows, cols), dtype=np.float64) for i in range(nz)]

    def setZero(self):
        for t1 in range(NLSQ_MAX_PARAM_TYPES):
            for t2 in range(NLSQ_MAX_PARAM_TYPES):
                for j in range(len( self.Hs[t1][t2])):
                    self.Hs[t1][t2][j].fill(0)
                  
def adjust_structure_and_motion(cams, Xs, measurements2d, correspondingView,
                     correspondingPoint, inlierThreshold,
                     clip_IRLS_Weights = False,
                     log_file = 'running_logs.txt'):
    """
    Run bundle adjustment from the given input data.
    """

    #Providing information about the parameters for the non-linear least squares optimizer
    paramDesc = NLSQ_ParamDesc(2) # Camera + Point
    paramDesc.dimension[0] = 6
    paramDesc.dimension[1] = 3
    paramDesc.count[0] = len(cams)
    paramDesc.count[1] = len(Xs)
    usedParamTypes = [0, 1]
    
    nMeasurements = measurements2d.shape[0]    
    correspondingParams = np.zeros((measurements2d.shape[0], 2), dtype=int)
    for k in range(nMeasurements):
        correspondingParams[k][0] = correspondingView[k]
        correspondingParams[k][1] = correspondingPoint[k]

    costFun = BundleCostFunction(paramDesc, usedParamTypes, inlierThreshold, cams, Xs,
                                 measurements2d, correspondingParams)
                        
    costFunctions = []
    costFunctions.append(costFun)    
    opt = NLSQ_Optimizer(paramDesc, costFunctions,                      \
                         clipIRLS_Weights=clip_IRLS_Weights ,log_file=log_file)

    opt.maxIterations = 10
    opt.minimize()
    


    #DEBUG: Write the final residuals to file
    with open("final_residuals.txt", "w") as f:
        for k in range(costFun.nMeasurements):
            res0 = costFun.residuals[k][0]
            res1 = costFun.residuals[k][1]
            f.write('{:.4f} {:.4f}\n'.format(res0, res1))

    
    
    
    
