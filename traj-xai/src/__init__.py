"""
traj-xai: A package for explainable AI on trajectory data.
"""

from .segmentation import (
    rdp_segmentation,
    random_segmentation,
    sliding_window_segmentation,
    mdl_segmentation
)

from .perturbation import (
    gaussian_perturbation,
    scaling_perturbation,
    rotation_perturbation
)

from .evaluation import ap_at_k

from .xai import TrajectoryManipulator

from .experiment import (
    experiment,
    run_experiments,
    save_result_row,
    check_ram_and_log,
    generate_unique_name
)

__all__ = [
    'rdp_segmentation',
    'random_segmentation',
    'sliding_window_segmentation',
    'mdl_segmentation',
    'gaussian_perturbation',
    'scaling_perturbation',
    'rotation_perturbation',
    'ap_at_k',
    'TrajectoryManipulator',
    'experiment',
    'run_experiments',
    'save_result_row',
    'check_ram_and_log',
    'generate_unique_name'
]