"""
Trajectory segmentation module.
"""

from .segmentation import _segmenter


# For backward compatibility
def rdp_segmentation(traj, epsilon=0.005):
    """Legacy function for RDP segmentation."""
    return _segmenter.apply(traj, "rdp", {"epsilon": epsilon})

def random_segmentation(traj, num_segments=4):
    """Legacy function for random segmentation."""
    return _segmenter.apply(traj, "random", {"num_segments": num_segments})

def sliding_window_segmentation(traj, step=5, percentage=5):
    """Legacy function for sliding window segmentation."""
    return _segmenter.apply(
        traj, "sliding_window", {"step": step, "percentage": percentage}
    )

def mdl_segmentation(traj, epsilon=0.8):
    """Legacy function for MDL segmentation."""
    return _segmenter.apply(traj, "mdl", {"epsilon": epsilon})

__all__ = [
    "rdp_segmentation",
    "random_segmentation",
    "sliding_window_segmentation",
    "mdl_segmentation"
]
