"""
Perturbation methods for trajectory manipulation.
"""
import numpy as np

def gaussian_perturbation(segment, mean=0, std=3, scale=1.5):
    """
    Apply Gaussian noise perturbation to a trajectory segment.
    
    Parameters:
        segment (list): List of trajectory points
        mean (float): Mean of Gaussian noise
        std (float): Standard deviation of Gaussian noise
        scale (float): Scale factor for noise magnitude
        
    Returns:
        list: Perturbed trajectory segment
    """
    new_segment = []
    for point in segment:
        x, y = point
        new_x = x + np.random.normal(mean, std) * scale
        new_y = y + np.random.normal(mean, std) * scale
        new_segment.append((new_x, new_y))
    return new_segment

def scaling_perturbation(segment, scale_factor=1.2):
    """
    Apply scaling perturbation to a trajectory segment.
    
    Parameters:
        segment (list): List of trajectory points
        scale_factor (float): Factor to scale the trajectory by
        
    Returns:
        list: Scaled trajectory segment
    """
    new_segment = []
    for point in segment:
        x, y = point
        new_x = x * scale_factor
        new_y = y * scale_factor
        new_segment.append((new_x, new_y))
    return new_segment

def rotation_perturbation(segment, angle=np.pi/18):
    """
    Apply rotation perturbation to a trajectory segment.
    
    Parameters:
        segment (list): List of trajectory points
        angle (float): Rotation angle in radians
        
    Returns:
        list: Rotated trajectory segment
    """
    new_segment = []
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    for point in segment:
        x, y = point
        new_x = x * cos_angle - y * sin_angle
        new_y = x * sin_angle + y * cos_angle
        new_segment.append((new_x, new_y))
    return new_segment