"""
Perturbation class for trajectory manipulation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class Perturbation:
    """
    A class for applying various perturbation methods to trajectory segments.

    This class provides a flexible way to apply one or multiple perturbation
    methods to trajectory segments with customizable parameters.
    """

    def __init__(self):
        """Initialize the Perturbation class."""
        self.available_methods = {
            "gaussian": self._gaussian_perturbation,
            "scaling": self._scaling_perturbation,
            "rotation": self._rotation_perturbation,
        }

        # Default parameters for each method
        self.default_params = {
            "gaussian": {"mean": 0, "std": 3, "scale": 1.5},
            "scaling": {"scale_factor": 1.2},
            "rotation": {"angle": np.pi / 18},
        }

    def _gaussian_perturbation(
        self,
        segment: List[Tuple[float, float]],
        mean: float = 0,
        std: float = 3,
        scale: float = 1.5,
    ) -> List[Tuple[float, float]]:
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

    def _scaling_perturbation(
        self, segment: List[Tuple[float, float]], scale_factor: float = 1.2
    ) -> List[Tuple[float, float]]:
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

    def _rotation_perturbation(
        self, segment: List[Tuple[float, float]], angle: float = np.pi / 18
    ) -> List[Tuple[float, float]]:
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

    def apply(
        self,
        segment: List[Tuple[float, float]],
        methods: List[str] = None,
        params: Dict[str, Dict[str, Any]] = None,
    ) -> List[Tuple[float, float]]:
        """
        Apply one or more perturbation methods to a trajectory segment.

        Parameters:
            segment (list): List of trajectory points
            methods (list): List of perturbation methods to apply
            params (dict): Dictionary of parameters for each method

        Returns:
            list: Perturbed trajectory segment

        Example:
            perturb = Perturbation()
            methods = ['gaussian', 'rotation']
            params = {
                'gaussian': {'mean': 0, 'std': 2, 'scale': 1.0},
                'rotation': {'angle': np.pi/4}
            }
            perturbed_segment = perturb.apply(segment, methods, params)
        """
        if methods is None:
            methods = ["gaussian"]

        if params is None:
            params = {}

        result = segment.copy()

        for method in methods:
            if method not in self.available_methods:
                raise ValueError(
                    f"Method '{method}' not found. Available methods: {list(self.available_methods.keys())}"
                )

            # Merge default parameters with custom parameters
            method_params = self.default_params.get(method, {}).copy()
            if method in params:
                method_params.update(params[method])

            result = self.available_methods[method](result, **method_params)

        return result

    def get_available_methods(self) -> List[str]:
        """
        Get list of available perturbation methods.

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

# Create an instance for legacy functions
_perturbation = Perturbation()
