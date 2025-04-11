"""
Evaluation metrics for trajectory explanation models.
"""


def ap_at_k(y_true, relevant_class, k):
    """
    Calculate AP@K (Average Precision at K) for a given class relevance.

    Parameters:
        y_true (list): List of class labels (e.g., ['c1', 'c1', 'c2', 'c1', 'c1', 'c2'])
        relevant_class (str): The class to consider as relevant (e.g., 'c1')
        k (int): The cut-off rank to consider for AP@K

    Returns:
        float: Average Precision at K
    """
    if k > len(y_true):
        k = len(y_true)  # Adjust k if it's larger than the list length

    num_relevant = 0  # Count of relevant items encountered so far
    score_sum = 0.0  # Sum of precision at each relevant point

    for i in range(k):
        if str(y_true[i]) != relevant_class:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            score_sum += precision_at_i

    ap_k = score_sum / min(num_relevant, k) if num_relevant > 0 else 0.0
    return ap_k
