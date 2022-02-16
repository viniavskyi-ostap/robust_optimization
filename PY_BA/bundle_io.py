#Hanling reading and writing files
import numpy as np
import bz2
from geometry import Camera, createRotationMatrixRodrigues

def readBundleData(file_name):
    """
    Read data from file
    """
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(int, file.readline().split())        
        correspondingView = np.empty(n_observations, dtype = int)
        correspondingPoint  = np.empty(n_observations, dtype = int)
        measurements      = np.empty((n_observations, 2))
        cams = [Camera() for i in range(n_cameras)]

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            correspondingView[i] = int(camera_index)
            correspondingPoint[i] = int(point_index)
            measurements[i] = [float(x), float(y)]
        
        for i in range(n_cameras):
            cam_params = np.zeros(9)
            for j in range(9):                
                cam_params[j]  = float(file.readline())
            omega = cam_params[0:3]
            
            cams[i].R = createRotationMatrixRodrigues(omega)
            cams[i].T = cam_params[3:6]

            f = cam_params[6]            
            cams[i].setIntrinsicMatrixFromF(f)
            k1 = cam_params[7]
            k2 = cam_params[8]
            f2 = f * f
            cams[i].distortion.setKParams(k1* f2, k2 * f2 * f2)
            
            
        Xs = np.empty((n_points,3))
        for i in range(n_points):            
            for j in range(3):
                Xs[i][j] = float(file.readline())        
            

    return cams, Xs, correspondingView, correspondingPoint, measurements


# def read_bal_data(file_name):
#     """
#     Read data from file
#     """
#     with bz2.open(file_name, "rt") as file:
#         n_cameras, n_points, n_observations = map(int, file.readline().split())
#         camera_indices = np.empty(n_observations, dtype = int)
#         point_indices  = np.empty(n_observations, dtype = int)
#         points_2d      = np.empty((n_observations, 2))

#         for i in range(n_observations):
#             camera_index, point_index, x, y = file.readline().split()
#             camera_indices[i] = int(camera_index)
#             points_2d[i] = [float(x), float(y)]

#         camera_params = np.empty(n_cameras*9)
#         for i in range(n_cameras*9):            
#             camera_params[i] = float(file.readline())

#         camera_params = camera_params.reshape((n_cameras, -1))

#         points_3d = np.empty(n_points * 3)
#         for i in range(n_points * 3):
#             points_3d[i] = float(file.readline())

#         points_3d = points_3d.reshape((n_points, -1))

#     return camera_params, points_3d, camera_indices, point_indices, points_2d

 
