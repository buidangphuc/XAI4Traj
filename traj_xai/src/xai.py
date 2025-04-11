"""
XAI methods for trajectory explanations.
"""

import random

import numpy as np
from fastdtw import fastdtw
from pactus import Dataset
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LogisticRegression
from yupi import Trajectory


class TrajectoryManipulator:
    """
    Manipulates trajectories to explain model predictions using XAI techniques.

    This class implements a method to explain black-box models by perturbing
    trajectory segments and analyzing the effect on model predictions.
    """

    def __init__(
        self,
        X,
        segmentation_model,
        perturbation_model,
        model,
        deep_learning_model=False,
    ):
        """
        Initialize the TrajectoryManipulator.

        Parameters:
            X (list): The trajectory points
            segmentation_model (callable): Function to segment trajectories
            perturbation_model (callable): Function to perturb trajectory segments
            model: The black box model to explain
            deep_learning_model (bool): Whether the model is a deep learning model
        """
        try:
            self.X = list(X)  # Convert to list instead of numpy array

            self.segmentation_model = segmentation_model
            self.perturbation_model = perturbation_model
            self.model = model
            self.deep_learning_model = deep_learning_model

            try:
                self.segments = self._segmentation(self.X)
            except Exception as e:
                print(f"Error in _segmentation: {e}")

            self.x_len = len(self.segments)
            self.number_of_permutations = min(2**10, 2**self.x_len)

            self.perturb_vectors = self.create_perturbation_points_by_shuffle(
                self.x_len, self.number_of_permutations
            )

            try:
                self.clean_segments = self.segments
                self.noisy_segments = [self._perturbation(seg) for seg in self.segments]
            except Exception as e:
                print(f"Error in creating noisy segments: {e}")

            try:
                self.Z_eval = self._createZForEval()
            except Exception as e:
                print(f"Error in _createZForEval: {e}")

        except Exception as e:
            print(f"Error in __init__: {e}")

    def _segmentation(self, points_list):
        """
        Segment trajectory using provided segmentation model.

        Parameters:
            points_list (list): List of trajectory points

        Returns:
            list: Segmented trajectory
        """
        try:
            return self.segmentation_model(points_list)
        except Exception as e:
            print(f"Error in _segmentation: {e}")
            return []

    def _perturbation(self, segment):
        """
        Apply perturbation to a trajectory segment.

        Parameters:
            segment (list): Trajectory segment to perturb

        Returns:
            list: Perturbed segment
        """
        try:
            perturbed_segment = self.perturbation_model(segment)
            return perturbed_segment
        except Exception as e:
            print(f"Error in _perturbation: {e}")
            return segment

    @staticmethod
    def create_perturbation_points_by_shuffle(vector_length, samples):
        """
        Create binary vectors for perturbation combinations.

        Parameters:
            vector_length (int): Length of perturbation vector
            samples (int): Number of samples to generate

        Returns:
            list: List of binary perturbation vectors
        """
        try:
            return [
                [random.randint(0, 1) for _ in range(vector_length)]
                for _ in range(samples)
            ]
        except Exception as e:
            print(f"Error in create_perturbation_points_by_shuffle: {e}")
            return [[0] * vector_length for _ in range(samples)]

    def _convert_perturb_vector_to_traj(self, vector):
        """
        Convert a perturbation vector to a trajectory.

        Parameters:
            vector (list): Binary perturbation vector

        Returns:
            list: Combined trajectory with perturbed segments
        """
        try:
            return sum(
                [
                    self.noisy_segments[i] if bit == 1 else self.clean_segments[i]
                    for i, bit in enumerate(vector)
                ],
                [],
            )
        except Exception as e:
            print(f"Error in _convert_perturb_vector_to_traj: {e}")
            return []

    def _perturbed_traj_generator(self):
        """
        Generate perturbed trajectories based on perturbation vectors.

        Yields:
            list: Perturbed trajectories
        """
        try:
            for vector in self.perturb_vectors:
                yield self._convert_perturb_vector_to_traj(vector)
        except Exception as e:
            print(f"Error in _perturbed_traj_generator: {e}")

    def _createZForEval(self):
        """
        Create evaluation trajectories with one segment perturbed at a time.

        Returns:
            list: List of trajectories for evaluation
        """
        try:
            identity_matrix = [
                [1 if i == j else 0 for j in range(self.x_len)]
                for i in range(self.x_len)
            ]
            return [
                sum(
                    [
                        self.noisy_segments[i] if bit == 1 else self.clean_segments[i]
                        for i, bit in enumerate(row)
                    ],
                    [],
                )
                for row in identity_matrix
            ]
        except Exception as e:
            print(f"Error in _createZForEval: {e}")
            return []

    def calc_dtw(self, raw):
        """
        Calculate DTW distance between raw trajectory and perturbed trajectories.

        Parameters:
            raw (list): Raw trajectory

        Returns:
            list: List of DTW distances
        """
        try:
            return [
                fastdtw(raw, pert, dist=euclidean)[0]
                for pert in self._perturbed_traj_generator()
            ]
        except Exception as e:
            print(f"Error in calc_dtw: {e}")
            return []

    def _calculate_weight(self):
        """
        Calculate weights for the logistic regression based on DTW distances.

        Returns:
            list: List of weights
        """
        try:
            distances = self.calc_dtw(self.X)
            mean_dist = sum(distances) / len(distances)
            std_distances = (
                sum((x - mean_dist) ** 2 for x in distances) / len(distances)
            ) ** 0.5
            weights = [
                1
                if std_distances == 0
                else (2.718 ** (-abs((d - mean_dist) / (std_distances + 1e-10))))
                for d in distances
            ]
            return weights
        except Exception as e:
            print(f"Error in _calculate_weight: {e}")
            return []

    def explain(self):
        """
        Generate explanations for trajectory classification.

        Returns:
            numpy.ndarray: Model coefficients for each class
        """
        try:
            Z_trajs = [
                Trajectory(points=Z_traj) for Z_traj in self._perturbed_traj_generator()
            ]
            labels = [1] * (len(Z_trajs) - 1) + [0]
            Z_pro = Dataset("custom", Z_trajs, labels)
            preds = self.model.predict(Z_pro)
            pred_labels = [pred.argmax() for pred in preds]
            Y = self.model.encoder.inverse_transform(pred_labels)

            # Check if Y only has one value, return None
            if len(np.unique(Y)) == 1:
                return None

            clf = LogisticRegression()
            clf.fit(self.perturb_vectors, Y, sample_weight=self._calculate_weight())

            self.coef_ = clf.coef_
            self.classes_ = clf.classes_
            return self.coef_
        except Exception as e:
            print(f"Error in explain: {e}")
            return np.zeros((1, self.x_len))

    def get_Y_eval_sorted(self):
        """
        Get sorted evaluation predictions.

        Returns:
            list: Sorted predictions
        """
        try:
            Z_trajs = [Trajectory(points=Z_traj) for Z_traj in self.Z_eval]
            labels = [1] * (len(Z_trajs) - 1) + [0]
            Z_pro = Dataset("custom1", Z_trajs, labels)
            Y = self.model.predict(Z_pro)  # Prediction results

            if Y is None:
                return None  # Return None if Y is None

            Y_without_pertub = self.get_Y()  # Actual result, single class

            class_index = None
            if len(np.unique(Y)) > 2:
                # Find index of class in self.classes_
                class_index = np.where(self.classes_ == Y_without_pertub[0])[0][0]

                # Get weights row for corresponding class from self.coef_
                class_coef = self.coef_[class_index]
            else:
                class_coef = self.coef_[0]

            # Sort indices by weight from high to low
            sorted_indices = np.argsort(abs(class_coef))[::-1]

            return [Y[i] for i in sorted_indices]

        except Exception as e:
            print(f"Error in get_Y_eval_sorted: {e}")
            return np.zeros(len(self.Z_eval))

    def get_Y(self):
        """
        Get model prediction for the original trajectory.

        Returns:
            list: Model prediction
        """
        try:
            Z_trajs = [Trajectory(points=self.X)]
            labels = [0]
            Z_pro = Dataset("custom1", Z_trajs, labels)
            Y = self.model.predict(Z_pro)
            return Y
        except Exception as e:
            print(f"Error in get_Y: {e}")
            return []

    def get_segment(self):
        """
        Get trajectory segments.

        Returns:
            list: List of segments
        """
        try:
            return self.segments
        except Exception as e:
            print(f"Error in get_segment: {e}")
            return []
