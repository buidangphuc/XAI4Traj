"""
Experiment execution logic for trajectory XAI.
"""
import os
import csv
import hashlib
import psutil
from datetime import datetime

from .xai import TrajectoryManipulator
from .evaluation import ap_at_k

def generate_unique_name(traj_points):
    """
    Generate a unique trajectory name based on the hash of trajectory points.
    
    Parameters:
        traj_points (list): List of trajectory points
        
    Returns:
        str: Unique trajectory name
    """
    try:
        traj_hash = hashlib.md5(str(traj_points).encode()).hexdigest()
        return f"traj_{traj_hash[:8]}"
    except Exception as e:
        print(f"Error generating unique name: {e}")
        return "traj_error"

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
                trajectory_experiment = TrajectoryManipulator(traj_points, segment_func, perturbation_func, blackbox_model)
            except Exception as e:
                print(f"Error initializing TrajectoryManipulator for trajectory {traj_idx}: {e}")
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
                    print(f"[DEBUG] Prediction failed for trajectory {traj_idx}. Skipping...")
                    continue
            except Exception as e:
                print(f"Error getting ground truth output for trajectory {traj_idx}: {e}")
                continue

            try:
                y_true = trajectory_experiment.get_Y_eval_sorted()
                if y_true is None:
                    print(f"Failed to retrieve label for trajectory {traj_idx}. Skipping...")
                    continue
            except Exception as e:
                print(f"Error retrieving perturbed output for trajectory {traj_idx}: {e}")
                continue

            try:
                y_true_array = y_true[0]
                change = 1 if any(item not in relevant_class for item in y_true) else 0
            except Exception as e:
                print(f"Error computing change for trajectory {traj_idx}: {e}")
                continue

            try:
                precision_score = ap_at_k(y_true, relevant_class, len(y_true)) if change else 0.0
            except Exception as e:
                print(f"Error computing precision score for trajectory {traj_idx}: {e}")
                precision_score = 0.0
                
            yield traj_idx, traj_name, change, precision_score
        except Exception as e:
            print(f"Unexpected error processing trajectory {traj_idx}: {e}")
            continue

def check_ram_and_log(ram_limit=28, log_dir='logs'):
    """
    Check RAM usage and log the result.
    
    Parameters:
        ram_limit (int): RAM usage limit percentage
        log_dir (str): Directory for log files
        
    Returns:
        bool: True if RAM usage exceeds the limit
    """
    os.makedirs(log_dir, exist_ok=True)
    ram_usage = psutil.virtual_memory().percent
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    log_message = f"{timestamp} - Total RAM: {total_ram:.2f} GB | Used: {ram_usage}%\n"
    log_file = os.path.join(log_dir, 'ram_usage_log.txt')
    with open(log_file, 'a') as f:
        f.write(log_message)

    print(log_message.strip())
    return ram_usage > ram_limit

def save_result_row(row, file_path):
    """
    Save a single row of results to a CSV file.
    
    Parameters:
        row (list): Row data to save
        file_path (str): Path to the CSV file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Initialize a set to track written rows (for deduplication)
    existing_rows = set()

    # Check if file exists
    if os.path.exists(file_path):
        # Read existing rows to prevent duplication
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for existing_row in reader:
                existing_rows.add(tuple(existing_row))  # Convert row to tuple
    else:
        # File does not exist; create it with a header
        with open(file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'traj_name', 'change', 'precision_score'])

    # Check if the row is already in the file
    if tuple(row) in existing_rows:
        print(f"[INFO] Row already exists in {file_path}: {row}")
        return

    # Append the row to the file
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        print(f"[INFO] Row saved to {file_path}: {row}")

def run_experiments(dataset, segment_funcs, perturbation_funcs, model, log_dir='logs'):
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
            file_path = os.path.join(log_dir, f"{segment_func.__name__}_{perturbation_func.__name__}_results.csv")
            
            # Run the experiment
            print(f"Running experiment with {segment_func.__name__} and {perturbation_func.__name__}")
            
            # Loop through the experiment results and save row by row
            for result in experiment(dataset, segment_func, perturbation_func, model):
                traj_idx, traj_name, change, precision_score = result
                
                # Save each row to the CSV
                save_result_row(
                    [traj_idx, traj_name, change, precision_score],
                    file_path
                )
                
                # Check RAM usage periodically
                if traj_idx % 10 == 0:
                    if check_ram_and_log(ram_limit=80, log_dir=log_dir):
                        print("RAM usage too high. Pausing experiment...")
                        break
                        
            print(f"Results saved to {file_path}")