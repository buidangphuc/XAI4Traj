import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder

import pactus.config as cfg
from pactus import Dataset
from pactus.dataset import Data
from pactus.models import Model
from pactus.models.evaluation import Evaluation

NAME = "hmm_model"

class HMMModel(Model):
    """Implementation of a Hidden Markov Model for trajectory classification."""

    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = 'diag',
        n_iter: int = 100,
        metrics=None,
        random_state: Union[int, None] = None,
    ):
        super().__init__(NAME)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.metrics = ["accuracy"] if metrics is None else metrics
        self.encoder: Union[LabelEncoder, None] = None
        self.labels: Union[List[Any], None] = None
        self.models: Dict[int, hmm.GaussianHMM] = {}
        self.random_state = random_state
        self.set_summary(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            metrics=self.metrics,
        )

    def train(
        self,
        data: Data,
        original_data: Data,
        training=True,  # For compatibility with parent class
        **kwargs  # Not used but included for compatibility
    ):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            logging.warning(
                f"Custom seed provided for {self.name} model. This "
                "sets random seeds for python and numpy."
            )
        
        self.encoder = None
        self.labels = data.labels
        
        # Encode the labels
        y_encoded = self._encode_labels(data)
        
        # Extract trajectory features
        x_trajs = self._extract_features(data)
        
        # Train one HMM per class
        self._train_hmm_models(x_trajs, y_encoded)
        
        logging.info(f"Trained HMM models for {len(self.models)} classes")

    def predict(self, data: Data) -> List[Any]:
        """
        Predict class probabilities for each trajectory
        
        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        assert self.models is not None, "Model has not been trained yet"
        assert self.encoder is not None, "Encoder is not initialized"

        x_trajs = self._extract_features(data)
        n_samples = len(x_trajs)
        n_classes = len(self.encoder.classes_)
        
        # Initialize probabilities array
        probabilities = np.zeros((n_samples, n_classes))
        
        for i, features in enumerate(x_trajs):
            # Calculate log-likelihood for each class
            log_likelihoods = np.array([
                model.score(features) for class_idx, model in self.models.items()
            ])
            
            # Convert log-likelihoods to probabilities
            # Use softmax to normalize
            log_likelihoods = np.array(log_likelihoods)
            log_likelihoods -= np.max(log_likelihoods)  # For numerical stability
            probabilities[i] = np.exp(log_likelihoods)
            probabilities[i] /= np.sum(probabilities[i])

        return probabilities

    def evaluate(self, data: Data) -> Evaluation:
        """Evaluate the model on test data"""
        assert self.encoder is not None, "Encoder is not set."
        
        # Get predicted probabilities
        probabilities = self.predict(data)
        
        # Get class with highest probability for each sample
        pred_indices = np.argmax(probabilities, axis=1)
        
        # Convert back to original class labels
        predictions = self.encoder.inverse_transform(pred_indices)
        
        return Evaluation.from_data(data, predictions, self.summary)

    def _encode_labels(self, data: Data) -> np.ndarray:
        """Encode the labels as integers"""
        if self.encoder is None:
            self.encoder = LabelEncoder()
            self.encoder.fit(self.labels)
        encoded_labels = self.encoder.transform(data.labels)
        assert isinstance(encoded_labels, np.ndarray)
        return encoded_labels

    def _extract_features(self, data: Data) -> List[np.ndarray]:
        """
        Extract features from trajectories for HMM modeling
        
        Returns:
            List of feature arrays, one per trajectory
        """
        features_list = []
        for traj in data.trajs:
            # Extract coordinates and time
            coords = traj.r
            times = traj.t.reshape(-1, 1)
            
            # Combine them into a single array [x, y, t]
            traj_data = np.hstack((coords, times))
            
            # Process into HMM features 
            features = self._preprocess_trajectory(traj_data)
            features_list.append(features)
            
        return features_list

    def _preprocess_trajectory(self, traj: np.ndarray) -> np.ndarray:
        """
        Process trajectory data into features for HMM
        
        Input: traj = array of shape (n, 3) with columns [x, y, t]
        Output: features array of shape (n-1, 2) with columns [dx, dy]
        """
        dx = np.diff(traj[:, 0])
        dy = np.diff(traj[:, 1])
        
        # Combine into feature array
        features = np.stack([dx, dy], axis=1)
        return features

    def _train_hmm_models(self, x_trajs: List[np.ndarray], y_encoded: np.ndarray) -> None:
        """
        Train 1 HMM for each class label.
        
        Args:
            x_trajs: List of preprocessed trajectory features
            y_encoded: Array of encoded class labels
        """
        from collections import defaultdict

        # Group trajectories by class
        class_trajs = defaultdict(list)
        for features, label in zip(x_trajs, y_encoded):
            class_trajs[label].append(features)

        # Train one model per class
        for label, feats_list in class_trajs.items():
            # Combine all sequences for this class
            X = np.vstack(feats_list)
            lengths = [len(f) for f in feats_list]

            # Train the model
            model = hmm.GaussianHMM(
                n_components=self.n_components, 
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state
            )
            
            try:
                model.fit(X, lengths)
                self.models[label] = model
                logging.info(f"Trained model for class {label} with data shape {X.shape}")
            except Exception as e:
                logging.error(f"Failed to train model for class {label}: {str(e)}")

