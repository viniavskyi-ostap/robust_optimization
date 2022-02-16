# Very simple test for linear fitting using Distribution Matching
import cupy as np
import math
from robust_kernels import *
import matplotlib.pyplot as plt
import pdb

def gen_data(N, d, outlier_rate = 0.5, noise_variance = 1.0, outlier_variance = 50.0):
    """
    Generate a random set containing N data points, each of dimension d
    """
    X = np.random.rand(N,d-1)
    X = np.concatenate((X, np.ones((N,1))), axis = 1)
    theta = np.random.rand(d,1)
    y = X.dot(theta) + noise_variance*np.random.randn(N,1)
    y = y.ravel()
    #Generate outliers by corrupting the output:
    for i in range(int(N*outlier_rate)):
        y[i] += outlier_variance*np.random.randn()
    
    return X,  y, theta


def count_inliers(X, y, theta, inlier_threshold = 1.0):
    """
    Count the number of inliers given the data and current estimate theta
    """
    residuals = [y[i] - X[i,:].dot(theta) for i in range(X.shape[0])]
    inlier_count = 0
    for i in range(X.shape[0]):
        if abs(residuals[i]) <= inlier_threshold:
            inlier_count += 1
    return inlier_count


def eval_cost(X, y, theta):
    residual = [(X[i,:].dot(theta) - y[i])**2 for i in range(X.shape[0])]
    psi = Psi_SmoothTrunc()
    cost = 0
    for i in range(len(residual)):
        cost += psi.fun(residual[i])
    return cost

    
    
def optimal_transport_fit(X, y, true_residuals, theta0 = None, max_iter = 100, inlier_threshold = 1.0):
    """
    Iteratively conduct residual matching and parameter estimation
    Assume that we are given true_sorted residuals, we first match 
    the sorted residuals of the current estimate to the true sorted residuals
    then optimize the parameters. The process repeats until convergence
    """
    sorted_residuals = np.sort(true_residuals)
    theta = theta0 # Start from initialize theta

    #Now, need to extract out only inliers
    inls_residuals = [sorted_residuals[i] for i in range(len(sorted_residuals)) if abs(sorted_residuals[i])< inlier_threshold]
    for iteration in range(max_iter):
        
        #Re-evaluate the current number of inliers:
        inliers = count_inliers(X, y, theta)
        cost = eval_cost(X, y, theta)
        print("Current Iteration = {}, #inliers = {}, cost = {}".format(iteration, inliers, cost))


        #First, estimate the residuals for the current estimate
        estimated_residuals = [y[i] - X[i,:].dot(theta) for i in range(X.shape[0])]
        
        #Extract the data points with the smallest absolute residuals
        abs_res = [abs(estimated_residuals[i]) for i in range(len(estimated_residuals))]
        idx_sorted_abs_res = np.argsort(abs_res)

        num_inls = len(inls_residuals)
        extracted_idx = idx_sorted_abs_res[:num_inls]    
        extracted_residuals = [estimated_residuals[i] for i in extracted_idx]

        idx = np.argsort(extracted_residuals)
        idx2 = [extracted_idx[j] for j in idx]
        extracted_idx = idx2
        
        #Next, get the indices of the sorted residuals
        #sorted_indices = np.argsort(estimated_residuals)
        #sorted_values = np.sort(estimated_residuals)
        # for i in range(len(sorted_indices)): y_prime[sorted_indices[i]] = y[sorted_indices[i]] - sorted_residuals[i]

        #Generate a new vector y to solve least squares
        X_prime = X[extracted_idx,:] 
        y_prime = [] 
        for i in range(len(extracted_idx)):
            y_prime.append(y[extracted_idx[i]] - inls_residuals[i])
        #Now, solve least squares to update theta:
        theta = np.linalg.lstsq(X_prime, y_prime)[0]

       
    
#Main 
    
print("Testing Linear Fitting")
N = 5000
d = 500 
X, y, theta = gen_data(N,d, outlier_rate = 0.4)

#Try least squares fit:
lsq = np.linalg.lstsq(X, y)[0]
#count inliers:
gt_inliers = count_inliers(X, y, theta)
print("Ground truth inliers = ", gt_inliers)
lsq_inliers = count_inliers(X, y, lsq.reshape(X.shape[1],1))
print("Least Squares inliers = ", lsq_inliers)
true_residuals = np.array([y[i] - X[i,:].dot(theta) for i in range(X.shape[0])]).ravel()

sorted_true_residuals = np.sort(true_residuals)

random_theta = np.random.rand(len(lsq))
optimal_transport_fit(X, y, true_residuals, theta0 = lsq, max_iter = 550)
