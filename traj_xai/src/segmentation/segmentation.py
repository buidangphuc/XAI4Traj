"""
Trajectory segmentation algorithms for trajectory analysis.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rdp import rdp


class Segmentation:
    """
    A class for trajectory segmentation using various methods.

    This class provides a flexible way to apply different segmentation
    algorithms to trajectories with customizable parameters.
    """

    def __init__(self):
        """Initialize the Segmentation class."""
        # Available segmentation methods
        self.available_methods = {
            "rdp": self._rdp_segmentation,
            "random": self._random_segmentation,
            "sliding_window": self._sliding_window_segmentation,
            "mdl": self._mdl_segmentation,
        }

        # Default parameters for each method
        self.default_params = {
            "rdp": {"epsilon": 0.005},
            "random": {"num_segments": 4},
            "sliding_window": {"step": 5, "percentage": 5},
            "mdl": {"epsilon": 0.8},
        }

    def _rdp_segmentation(
        self, traj: List[Tuple[float, float]], epsilon: float = 0.005
    ) -> List[List[Tuple[float, float]]]:
        """
        Segment trajectory using the RDP algorithm while ensuring reduced points are used only once.

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
                    last_used_index = (
                        i  # Update the index of the last used reduced point
                    )

            # Append the last segment if it exists
            if current_segment:
                segments.append(current_segment)

            return segments
        except Exception as e:
            print(f"Error: {e}")
            return []

    def _random_segmentation(
        self, traj: List[Tuple[float, float]], num_segments: int = 4
    ) -> List[List[Tuple[float, float]]]:
        """
        Segment trajectory into a random number of segments.

        Parameters:
            traj (list): List of trajectory points
            num_segments (int): Number of segments to create

        Returns:
            list: List of trajectory segments
        """
        if len(traj) <= num_segments:
            return [[point] for point in traj]

        indices = sorted(random.sample(range(1, len(traj)), num_segments - 1))
        segments = [traj[i:j] for i, j in zip([0] + indices, indices + [len(traj)])]
        return segments

    def _sliding_window_segmentation(
        self, traj: List[Tuple[float, float]], step: int = 5, percentage: float = 5
    ) -> List[List[Tuple[float, float]]]:
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
        segments = [
            traj[i : i + window_size]
            for i in range(0, len(traj) - window_size + 1, step)
        ]
        return segments

    @staticmethod
    def _mdl_cost(segment: List[Tuple[float, float]]) -> float:
        """
        Calculate MDL cost for a trajectory segment.

        Parameters:
            segment (list): List of points in segment

        Returns:
            float: MDL cost
        """
        if len(segment) < 2:
            return float("inf")  # Cannot describe with fewer than 2 points

        start, end = segment[0], segment[-1]
        segment_array = np.array(segment)
        line = np.array(end) - np.array(start)

        if np.linalg.norm(line) == 0:
            return float("inf")  # Avoid division by zero if start == end

        normalized_line = line / np.linalg.norm(line)
        projections = np.dot(segment_array - start, normalized_line)
        reconstruction = start + np.outer(projections, normalized_line)
        error = np.linalg.norm(segment_array - reconstruction, axis=1).sum()

        return error

    def _mdl_segmentation(
        self, traj: List[Tuple[float, float]], epsilon: float = 0.8
    ) -> List[List[Tuple[float, float]]]:
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
            if self._mdl_cost(current_segment) > epsilon:
                # Close previous segment
                segments.append(current_segment[:-1])
                current_segment = [traj[i]]

        # Add final segment
        if current_segment:
            segments.append(current_segment)

        return segments

    def apply(
        self,
        traj: List[Tuple[float, float]],
        method: str = "rdp",
        params: Dict[str, Any] = None,
    ) -> List[List[Tuple[float, float]]]:
        """
        Apply a segmentation method to a trajectory.

        Parameters:
            traj (list): List of trajectory points
            method (str): The segmentation method to use
            params (dict): Parameters for the segmentation method

        Returns:
            list: List of trajectory segments

        Example:
            segmenter = Segmentation()
            segments = segmenter.apply(trajectory, 'rdp', {'epsilon': 0.01})
        """
        if method not in self.available_methods:
            raise ValueError(
                f"Method '{method}' not found. Available methods: {list(self.available_methods.keys())}"
            )

        # Merge default parameters with custom parameters
        method_params = self.default_params.get(method, {}).copy()
        if params is not None:
            method_params.update(params)

        return self.available_methods[method](traj, **method_params)

    def get_available_methods(self) -> List[str]:
        """
        Get list of available segmentation methods.

        Returns:
            list: List of available method names
        """
        return list(self.available_methods.keys())

    def get_default_params(
        self, method: Optional[str] = None
    ) -> Union[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Get default parameters for specified method or all methods.

        Parameters:
            method (str, optional): The method to get parameters for

        Returns:
            dict: Default parameters
        """
        if method is None:
            return self.default_params

        if method not in self.available_methods:
            raise ValueError(
                f"Method '{method}' not found. Available methods: {list(self.available_methods.keys())}"
            )

        return self.default_params[method]
    
# Create instance for legacy functions
_segmenter = Segmentation()
