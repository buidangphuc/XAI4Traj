import numpy as np

def preprocess_trajectory(traj):
    """
    Input: traj = array of shape (n, 3) with columns [x, y, t]
    Output: features array of shape (n-1, 4) with columns [dx, dy, vx, vy]
    """
    traj = np.array(traj)
    dx = np.diff(traj[:, 0])
    dy = np.diff(traj[:, 1])
    dt = np.diff(traj[:, 2])
    dt[dt == 0] = 1e-5  # tránh chia cho 0

    features = np.stack([dx, dy], axis=1)
    return features

from hmmlearn import hmm

def train_hmm_models(X_trajs, y_labels, n_components=4):
    """
    Train 1 HMM for each class label.

    X_trajs: list of trajectories (each trajectory is array of shape (n_i, 3) -> (x, y, t))
    y_labels: list of labels (same length as X_trajs)

    Return: dict {label: trained_hmm_model}
    """
    from collections import defaultdict

    class_trajs = defaultdict(list)
    for traj, label in zip(X_trajs, y_labels):
        features = preprocess_trajectory(traj)
        class_trajs[label].append(features)

    models = {}
    for label, feats_list in class_trajs.items():
        X = np.vstack(feats_list)
        lengths = [len(f) for f in feats_list]

        model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=100)
        model.fit(X, lengths)
        models[label] = model

    return models

def predict_trajectory(models, traj):
    """
    Predict the label of a single trajectory.

    models: dict of trained HMMs {label: model}
    traj: array of shape (n, 3) with (x, y, t)

    Return: predicted label
    """
    features = preprocess_trajectory(traj)
    scores = {label: model.score(features) for label, model in models.items()}
    return max(scores, key=scores.get)

def convert(data):
  res = []
  for traj in data:
    coor = traj.r
    time = traj.t
    coor = coor.tolist()
    time = time.tolist()
    combined_list = [l1 + [l2] for l1, l2 in zip(coor, time)]
    res.append(combined_list)
  return res

def convert_lablel(data):
  gt_lst = []
  for traj in data:
    gt = ds.labels[int(traj.traj_id)]
    gt_lst.append(gt)
  return gt_lst

# Example usage (wrapped in a function to avoid immediate execution)
def run_hmm_example(train, test, ds):
    from sklearn.preprocessing import LabelEncoder
    
    train_data = convert(train.trajs)
    test_data = convert(test.trajs)

    train_label = convert_lablel(train.trajs)
    test_label = convert_lablel(test.trajs)

    sequences = train_data
    le = LabelEncoder()
    le.fit(train_label)
    labels = le.transform(train_label)
    test_labels = le.transform(test_label)

    models = train_hmm_models(train_data, labels)
    # Predict for new trajectory
    y_pred = predict_trajectory(models, test_data[0])
    print("Predicted class:", y_pred)

    count = 0
    for i in range(len(test_data)):
        pred = predict_trajectory(models, test_data[i])
        gt = test_labels[i]
        if pred == gt:
            count = count + 1
    
    accuracy = count / len(test_data)
    print(f"Accuracy: {accuracy:.4f} ({count}/{len(test_data)})")
    
    return models, accuracy