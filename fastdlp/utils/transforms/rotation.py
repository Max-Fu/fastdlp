"""Rotation transform utilities for 6D rotation representation.

This module provides functions to convert between different rotation representations:
- 6D rotation representation
- Rotation matrices
- Euler angles
- Quaternions
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Literal

def rot_mat_to_rot_6d(rot_mat: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to 6D representation.
    
    Args:
        rot_mat: Rotation matrix of shape (N, 3, 3)
        
    Returns:
        6D rotation representation of shape (N, 6)
    """
    rot_6d = rot_mat[:, :2, :]  # N, 2, 3
    return rot_6d.reshape(-1, 6)  # N, 6

def rot_6d_to_rot_mat(rot_6d: np.ndarray) -> np.ndarray:
    """Convert a 6D representation to rotation matrix.
    
    Args:
        rot_6d: 6D rotation representation of shape (N, 6)
        
    Returns:
        Rotation matrix of shape (N, 3, 3)
    """
    rot_6d = rot_6d.reshape(-1, 2, 3)
    # assert the first two vectors are orthogonal
    if not np.allclose(np.sum(rot_6d[:, 0] * rot_6d[:, 1], axis=-1), 0):
        rot_6d = gram_schmidt(rot_6d)

    rot_mat = np.zeros((rot_6d.shape[0], 3, 3))
    rot_mat[:, :2, :] = rot_6d
    rot_mat[:, 2, :] = np.cross(rot_6d[:, 0], rot_6d[:, 1])
    return rot_mat

def euler_to_rot_6d(euler: np.ndarray, format: str = "XYZ") -> np.ndarray:
    """Convert euler angles to 6D representation.
    
    Args:
        euler: Euler angles of shape (N, 3)
        format: Euler angle format (default: "XYZ")
        
    Returns:
        6D rotation representation of shape (N, 6)
    """
    rot_mat = Rotation.from_euler(format, euler, degrees=False).as_matrix()
    return rot_mat_to_rot_6d(rot_mat)

def rot_6d_to_euler(rot_6d: np.ndarray, format: str = "XYZ") -> np.ndarray:
    """Convert 6D representation to euler angles.
    
    Args:
        rot_6d: 6D rotation representation of shape (N, 6)
        format: Euler angle format (default: "XYZ")
        
    Returns:
        Euler angles of shape (N, 3)
    """
    rot_mat = rot_6d_to_rot_mat(rot_6d)
    return Rotation.from_matrix(rot_mat).as_euler(format, degrees=False)

def quat_to_rot_6d(quat: np.ndarray, format: str = "wxyz") -> np.ndarray:
    """Convert quaternion to 6D representation.
    
    Args:
        quat: Quaternion of shape (N, 4)
        format: Quaternion format, either "wxyz" or "xyzw" (default: "wxyz")
        
    Returns:
        6D rotation representation of shape (N, 6)
    """
    assert format in ["wxyz", "xyzw"], "Invalid quaternion format, only support wxyz or xyzw"
    if format == "wxyz":
        quat = quat[:, [1, 2, 3, 0]]
    rot_mat = Rotation.from_quat(quat).as_matrix()
    return rot_mat_to_rot_6d(rot_mat)

def rot_6d_to_quat(rot_6d: np.ndarray, format: str = "wxyz") -> np.ndarray:
    """Convert 6D representation to quaternion.
    
    Args:
        rot_6d: 6D rotation representation of shape (N, 6)
        format: Quaternion format, either "wxyz" or "xyzw" (default: "wxyz")
        
    Returns:
        Quaternion of shape (N, 4)
    """
    rot_mat = rot_6d_to_rot_mat(rot_6d)
    quat = Rotation.from_matrix(rot_mat).as_quat()
    if format == "wxyz":
        quat = quat[:, [3, 0, 1, 2]]
    return quat

def euler_to_quat(euler: np.ndarray, format_euler: str = "XYZ", format_quat: str = "wxyz") -> np.ndarray:
    """Convert euler angles to quaternion.
    
    Args:
        euler: Euler angles of shape (N, 3)
        format_euler: Euler angle format (default: "XYZ")
        format_quat: Quaternion format, either "wxyz" or "xyzw" (default: "wxyz")
        
    Returns:
        Quaternion of shape (N, 4)
    """
    assert format_quat in ["wxyz", "xyzw"], "Invalid quaternion format, only support wxyz or xyzw"
    quat = Rotation.from_euler(format_euler, euler, degrees=False).as_quat()
    if format_quat == "wxyz":
        quat = quat[:, [3, 0, 1, 2]]
    return quat

def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """Apply Gram-Schmidt process to a set of vectors.
    
    Args:
        vectors: Vectors of shape (batchsize, N, D)
        
    Returns:
        Orthonormal basis of shape (batchsize, N, D)
    """
    if len(vectors.shape) == 2:
        vectors = vectors[None]
    
    basis = np.zeros_like(vectors)
    basis[:, 0] = vectors[:, 0] / np.linalg.norm(vectors[:, 0], axis=-1, keepdims=True)
    for i in range(1, vectors.shape[1]):
        v = vectors[:, i]
        for j in range(i):
            v -= np.sum(v * basis[:, j], axis=-1, keepdims=True) * basis[:, j]
        basis[:, i] = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return basis

def calculate_delta_rot(euler_rot_start: np.ndarray, euler_rot_end: np.ndarray, format: str = "XYZ") -> np.ndarray:
    """Calculate the delta rotation between two euler angles.
    
    Args:
        euler_rot_start: Starting euler angles of shape (N, 3)
        euler_rot_end: Ending euler angles of shape (N, 3)
        format: Euler angle format (default: "XYZ")
        
    Returns:
        Delta rotation in euler angles of shape (N, 3)
    """
    r = Rotation.from_euler(format, euler_rot_start, degrees=False)
    r2 = Rotation.from_euler(format, euler_rot_end, degrees=False)
    delta_rot = r2 * r.inv()
    euler_rot = delta_rot.as_euler(format, degrees=False)
    return euler_rot 