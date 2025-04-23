"""
Experiment execution logic for trajectory XAI.
"""

import os

from .evaluation import ap_at_k
from .utils import check_ram_and_log, generate_unique_name, save_result_row
from .xai import TrajectoryManipulator


def experiment(dataset, segment_func, perturbation_func, blackbox_model):
    """
    Run experiment on a dataset using the specified segmentation and perturbation functions.

    Parameters:
        dataset: The dataset to run experiments on
        segment_func (callable): Function for trajectory segmentation
        perturbation_func (callable): Function for trajectory perturbation
        blackbox_model: The model to explain

    Yields:
        tuple: (trajectory_index, trajectory_name, change_flag, precision_score)
    """
    for traj_idx, (traj, label) in enumerate(zip(dataset.trajs, dataset.labels)):
        try:
            traj_points, traj_label = traj.r, label

            if traj_points is None or len(traj_points) == 0:
                print(f"Trajectory {traj_idx} is empty or None. Skipping...")
                continue

            traj_name = generate_unique_name(traj_points)

            try:
                trajectory_experiment = TrajectoryManipulator(
                    traj_points, segment_func, perturbation_func, blackbox_model
                )
            except Exception as e:
                print(
                    f"Error initializing TrajectoryManipulator for trajectory {traj_idx}: {e}"
                )
                continue

            try:
                coef = trajectory_experiment.explain()
                if coef is None:
                    print("Doesn't change classification")
                    yield traj_idx, traj_name, 0, 0.0
                    continue
            except Exception as e:
                print(f"Error explaining trajectory {traj_idx}: {e}")
                continue

            try:
                segments = trajectory_experiment.get_segment()
            except Exception as e:
                print(f"Error retrieving segments for trajectory {traj_idx}: {e}")
                continue

            try:
                relevant_class = trajectory_experiment.get_Y()
                if relevant_class is None:
                    print(
                        f"[DEBUG] Prediction failed for trajectory {traj_idx}. Skipping..."
                    )
                    continue
            except Exception as e:
                print(
                    f"Error getting ground truth output for trajectory {traj_idx}: {e}"
                )
                continue

            try:
                y_true = trajectory_experiment.get_Y_eval_sorted()
                if y_true is None:
                    print(
                        f"Failed to retrieve label for trajectory {traj_idx}. Skipping..."
                    )
                    continue
            except Exception as e:
                print(
                    f"Error retrieving perturbed output for trajectory {traj_idx}: {e}"
                )
                continue

            try:
                import numpy as np
                
                # Improved handling of different data types
                change = 0
                for item in y_true:
                    # Check if this prediction is in the relevant class set
                    is_in_relevant = False
                    
                    # Data type-aware comparison
                    for cls in relevant_class:
                        # Case 1: Both are numpy arrays
                        if isinstance(item, np.ndarray) and isinstance(cls, np.ndarray):
                            if np.array_equal(item, cls):
                                is_in_relevant = True
                                break
                        # Case 2: Both are numpy-like objects with shape attribute
                        elif hasattr(item, 'shape') and hasattr(cls, 'shape'):
                            try:
                                if np.all(item == cls):  # Element-wise comparison
                                    is_in_relevant = True
                                    break
                            except:
                                pass  # If comparison fails, continue to next check
                        # Case 3: Standard equality for other types (strings, numbers, etc.)
                        elif item == cls:
                            is_in_relevant = True
                            break
                    
                    # If this prediction isn't in the relevant class, we have a change
                    if not is_in_relevant:
                        change = 1
                        break
            except Exception as e:
                print(f"Error computing change for trajectory {traj_idx}: {e}")
                continue

            try:
                precision_score = (
                    ap_at_k(y_true, relevant_class, len(y_true)) if change else 0.0
                )
            except Exception as e:
                print(f"Error computing precision score for trajectory {traj_idx}: {e}")
                precision_score = 0.0

            yield traj_idx, traj_name, change, precision_score
        except Exception as e:
            print(f"Unexpected error processing trajectory {traj_idx}: {e}")
            continue


def run_experiments(dataset, segment_funcs, perturbation_funcs, model, log_dir="logs"):
    """
    Run multiple experiments with different segmentation and perturbation functions.

    Parameters:
        dataset: The dataset to run experiments on
        segment_funcs (list): List of segmentation functions
        perturbation_funcs (list): List of perturbation functions
        model: The model to explain
        log_dir (str): Directory for log files
    """
    os.makedirs(log_dir, exist_ok=True)

    # Loop through segmentation and perturbation functions
    for segment_func in segment_funcs:
        for perturbation_func in perturbation_funcs:
            # Generate file path for saving results
            file_path = os.path.join(
                log_dir,
                f"{segment_func.__name__}_{perturbation_func.__name__}_results.csv",
            )

            # Run the experiment
            print(
                f"Running experiment with {segment_func.__name__} and {perturbation_func.__name__}"
            )

            # Loop through the experiment results and save row by row
            for result in experiment(dataset, segment_func, perturbation_func, model):
                traj_idx, traj_name, change, precision_score = result

                # Save each row to the CSV
                save_result_row(
                    [traj_idx, traj_name, change, precision_score], file_path
                )

                # Check RAM usage periodically
                if traj_idx % 10 == 0:
                    if check_ram_and_log(ram_limit=80, log_dir=log_dir):
                        print("RAM usage too high. Pausing experiment...")
                        break

            print(f"Results saved to {file_path}")
