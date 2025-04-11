"""
Utility functions for trajectory XAI.
"""

import csv
import hashlib
import os
from datetime import datetime

import psutil


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


def check_ram_and_log(ram_limit=28, log_dir="logs"):
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
    total_ram = psutil.virtual_memory().total / (1024**3)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_message = f"{timestamp} - Total RAM: {total_ram:.2f} GB | Used: {ram_usage}%\n"
    log_file = os.path.join(log_dir, "ram_usage_log.txt")
    with open(log_file, "a") as f:
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
        with open(file_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for existing_row in reader:
                existing_rows.add(tuple(existing_row))  # Convert row to tuple
    else:
        # File does not exist; create it with a header
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "traj_name", "change", "precision_score"])

    # Check if the row is already in the file
    if tuple(row) in existing_rows:
        print(f"[INFO] Row already exists in {file_path}: {row}")
        return

    # Append the row to the file
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        print(f"[INFO] Row saved to {file_path}: {row}")
