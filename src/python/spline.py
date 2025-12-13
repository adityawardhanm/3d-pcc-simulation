import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
from pathlib import Path    
root = Path(__file__).resolve().parents[2]  # parent_directory


# LOAD CUDA SHARED LIBRARY
lib = ctypes.CDLL(str(root / "lib/fk.so"))
lib.generate_spline_points.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")
]
lib.generate_spline_points.restype = ctypes.c_int

lib.initialize_gpu_context.argtypes = [
    ctypes.c_int,                                      # num_segments
    ctypes.c_int,                                      # resolution
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),   # length
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")    # T_cumulative
]
lib.initialize_gpu_context.restype = ctypes.c_int

lib.update_spline_fast.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),   # kappa
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),   # theta
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),   # phi
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")    # output
]
lib.update_spline_fast.restype = ctypes.c_int

lib.update_transforms.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")    # T_cumulative
]
lib.update_transforms.restype = ctypes.c_int

lib.destroy_gpu_context.argtypes = []
lib.destroy_gpu_context.restype = ctypes.c_int


# FUNCTION TO COMPUTE TRANSFORMATION MATRIX FOR A SEGMENT - TBD (CHECK FORMULA AND STRAIGHT SEGMENT CASE)
def compute_transformation_matrix(kappa, theta, phi):

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Straight segment case - TBD
    if abs(kappa) < 1e-6:
        T = np.eye(4, dtype=np.float32)
        T[2, 3] = theta  
        return T
    
    T = np.zeros((4, 4), dtype=np.float32)          # Matrix initialization
    
    # Rotation part (top-left 3x3)
    T[0, 0] = cos_phi * cos_theta + sin_phi * sin_phi
    T[0, 1] = -sin_phi * cos_phi * (1 - cos_theta)
    T[0, 2] = cos_phi * sin_theta
    
    T[1, 0] = -sin_phi * cos_phi * (1 - cos_theta)
    T[1, 1] = sin_phi * sin_phi + cos_phi * cos_phi * cos_theta
    T[1, 2] = sin_phi * sin_theta
    
    T[2, 0] = -cos_phi * sin_theta
    T[2, 1] = -sin_phi * sin_theta
    T[2, 2] = cos_theta
    
    # Translation part (right column)
    T[0, 3] = cos_phi * (1 - cos_theta) / kappa
    T[1, 3] = sin_phi * (1 - cos_theta) / kappa
    T[2, 3] = sin_theta / kappa
    
    # Homogeneous coordinate
    T[3, 3] = 1.0
    
    return T

# FUNCTION TO PERFORM CUMULATIVE TRANSFORMATIONS 
def compute_cumulative_transforms(kappa, theta, phi, num_segments):

    if num_segments is None:
        num_segments = len(kappa)

    T_cumulative = np.zeros((num_segments, 4, 4), dtype=np.float32)
    T_cumulative[0] = np.eye(4, dtype=np.float32)

    T_prev = np.eye(4, dtype=np.float32)
    for i in range(1, num_segments):
        T_seg = compute_transformation_matrix(kappa[i-1], theta[i-1], phi[i-1])
        T_prev = T_prev @ T_seg
        T_cumulative[i] = T_prev

    return T_cumulative
