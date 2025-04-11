"""
Perturbation package for trajectory manipulation.
"""

from .perturbation import _perturbation
import numpy as np


# Convenience functions for backward compatibility
def gaussian_perturbation(segment, mean=0, std=3, scale=1.5):
    """Legacy function for Gaussian perturbation."""
    return _perturbation.apply(
        segment, ["gaussian"], {"gaussian": {"mean": mean, "std": std, "scale": scale}}
    )

def scaling_perturbation(segment, scale_factor=1.2):
    """Legacy function for scaling perturbation."""
    return _perturbation.apply(
        segment, ["scaling"], {"scaling": {"scale_factor": scale_factor}}
    )

def rotation_perturbation(segment, angle=np.pi / 18):
    """Legacy function for rotation perturbation."""
    return _perturbation.apply(segment, ["rotation"], {"rotation": {"angle": angle}})

__all__ = [
    "gaussian_perturbation",
    "scaling_perturbation",
    "rotation_perturbation",
]
