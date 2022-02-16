import numpy as np

#*****************************************************************
def project_to_SO3(R):
    u,_, vt = np.linalg.svd(R)
    return u.dot(vt)

#*****************************************************************
def makeCrossProductMatrix(v):
    """
    Return cross product of v to itself
    Input: v: ndarray 
    Output: v cross v
    """
    m = np.zeros((3,3), dtype=np.float64)
    m[0][0] = m[1][1] = m[2][2] = 0.0 
    m[0][1] = -v[2]
    m[0][2] = v[1]
    m[1][0] = v[2]    
    m[1][2] = -v[0]
    m[2][0] = -v[1]
    m[2][1] = v[0]
    return m

#*****************************************************************
def getVectorArray(count, paramDim, delta, rowStart):
    """
    Convert a vector to an array of vector, each containing the update parameters for a specific camera/point    
    """
    vectorArray = [np.zeros(paramDim) for i in range(count)]
    startPos = rowStart
    for i in range(count):
        vectorArray[i] = delta[startPos:startPos + (i+1)*paramDim]
        startPos += paramDim
    return vectorArray

#*****************************************************************



#------------------------------------------------------------------
class Logger:
    """
    A class to handle logs, write to file and 
    """
    def __init__(self, outputFile = 'running_log.txt'):
        self.cost_logs = []
        self.ouputFile = outputFile
        #Prepare the file
        file = open(outputFile, "w")
        file.close()

    def log(self, cost):
        self.cost_logs.append(cost)
        with open(self.ouputFile, "a") as file:
            file.write("{}\n".format(cost))


    
