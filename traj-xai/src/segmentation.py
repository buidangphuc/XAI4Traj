"""
Trajectory segmentation algorithms for trajectory analysis.
"""
import random
import numpy as np
from rdp import rdp

def rdp_segmentation(traj, epsilon=0.005):
    """
    Segment trajectory using the RDP algorithm while ensuring reduced points are used only once and no overlap occurs.
    
    Parameters:
        traj (list): List of trajectory points
        epsilon (float): Epsilon parameter for RDP algorithm
        
    Returns:
        list: List of trajectory segments
    """
    try:
        # Apply the RDP algorithm to get reduced points
        reduced_points = rdp(traj, epsilon=epsilon)
        
        # Convert reduced points to a set for easy lookup
        reduced_points_set = set(map(tuple, reduced_points))
        
        # Create segments based on reduced points
        segments = []
        current_segment = []
        
        # Track the index of the last used reduced point
        last_used_index = -1
        
        for i, point in enumerate(traj):
            current_segment.append(point)
            
            # If the point is a reduced point and it hasn't been used yet, create a segment
            if tuple(point) in reduced_points_set and i > last_used_index:
                segments.append(current_segment)
                current_segment = []
                last_used_index = i  # Update the index of the last used reduced point
        
        # Append the last segment if it exists
        if current_segment:
            segments.append(current_segment)
        
        return segments
    except Exception as e:
        print(f"Error: {e}")
        return None


def random_segmentation(traj, num_segments=4):
    """
    Segment trajectory into a random number of segments.
    
    Parameters:
        traj (list): List of trajectory points
        num_segments (int): Number of segments to create
        
    Returns:
        list: List of trajectory segments
    """
    indices = sorted(random.sample(range(1, len(traj)), num_segments - 1))
    segments = [traj[i:j] for i, j in zip([0] + indices, indices + [len(traj)])]
    return segments


def sliding_window_segmentation(traj, step=5, percentage=5):
    """
    Segment trajectory using sliding window approach.
    
    Parameters:
        traj (list): List of trajectory points
        step (int): Step size for sliding window
        percentage (float): Window size as percentage of trajectory length
        
    Returns:
        list: List of trajectory segments
    """
    if not 0 < percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100.")

    window_size = max(1, int((percentage / 100) * len(traj)))
    segments = [traj[i:i + window_size] for i in range(0, len(traj) - window_size + 1, step)]
    return segments


def mdl_cost(segment):
    """
    Calculate MDL cost for a trajectory segment.
    
    Parameters:
        segment (list): List of points in segment
        
    Returns:
        float: MDL cost
    """
    if len(segment) < 2:
        return float('inf')  # Cannot describe with fewer than 2 points

    start, end = segment[0], segment[-1]
    segment_array = np.array(segment)
    line = np.array(end) - np.array(start)

    if np.linalg.norm(line) == 0:
        return float('inf')  # Avoid division by zero if start == end

    normalized_line = line / np.linalg.norm(line)
    projections = np.dot(segment_array - start, normalized_line)
    reconstruction = start + np.outer(projections, normalized_line)
    error = np.linalg.norm(segment_array - reconstruction, axis=1).sum()

    return error


def mdl_segmentation(traj, epsilon=0.8):
    """
    Segment trajectory using MDL algorithm.
    
    Parameters:
        traj (list): List of trajectory points
        epsilon (float): Error threshold for MDL segmentation
        
    Returns:
        list: List of trajectory segments
    """
    segments = []
    current_segment = [traj[0]]

    for i in range(1, len(traj)):
        current_segment.append(traj[i])
        if mdl_cost(current_segment) > epsilon:
            # Close previous segment
            segments.append(current_segment[:-1])
            current_segment = [traj[i]]

    # Add final segment
    if current_segment:
        segments.append(current_segment)

    return segments