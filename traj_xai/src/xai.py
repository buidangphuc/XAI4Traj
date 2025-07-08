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
    """

    def __init__(
        self,
        X,
        segmentation_model,
        perturbation_model,
        model,
        deep_learning_model=False,
    ):
        try:
            self.X = list(X)
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
        try:
            return self.segmentation_model(points_list)
        except Exception as e:
            print(f"Error in _segmentation: {e}")
            return []

    def _perturbation(self, segment):
        try:
            return self.perturbation_model(segment)
        except Exception as e:
            print(f"Error in _perturbation: {e}")
            return segment

    @staticmethod
    def create_perturbation_points_by_shuffle(vector_length, samples):
        try:
            return [
                [random.randint(0, 1) for _ in range(vector_length)]
                for _ in range(samples)
            ]
        except Exception as e:
            print(f"Error in create_perturbation_points_by_shuffle: {e}")
            return [[0] * vector_length for _ in range(samples)]

    def _convert_perturb_vector_to_traj(self, vector):
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
        try:
            for vector in self.perturb_vectors:
                yield self._convert_perturb_vector_to_traj(vector)
        except Exception as e:
            print(f"Error in _perturbed_traj_generator: {e}")

    def _createZForEval(self):
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
        try:
            return [
                fastdtw(raw, pert, dist=euclidean)[0]
                for pert in self._perturbed_traj_generator()
            ]
        except Exception as e:
            print(f"Error in calc_dtw: {e}")
            return []

    def _calculate_weight(self):
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

    def _safe_predict(self, dataset, fallback_value=None):
        """
        Safely call model.predict with error handling for format mismatches.
        
        Args:
            dataset: The dataset to predict on
            fallback_value: Value to return if prediction fails
            
        Returns:
            Prediction results or fallback value
        """
        try:
            return self.model.predict(dataset)
        except ValueError as e:
            if "not enough values to unpack" in str(e):
                try:
                    if hasattr(dataset, 'trajs'):
                        return self.model.predict(dataset.trajs)
                    elif hasattr(self.model, 'predict_raw'):
                        return self.model.predict_raw(dataset)
                    elif hasattr(self.model, '_get_input_data'):
                        try:
                            x_data, y_data = self.model._get_input_data(dataset)
                            lengths = [len(x) for x in x_data]
                            return self.model._predict_with_data(x_data, lengths)
                        except Exception:
                            pass
                    # Try direct prediction with trajectory points
                    if isinstance(dataset, Dataset) and hasattr(dataset, 'trajs'):
                        try:
                            return self.model.predict([t.points for t in dataset.trajs])
                        except Exception:
                            # Try one more approach for neural networks
                            if self.deep_learning_model and hasattr(self.model, 'predict_proba'):
                                return self.model.predict_proba([t.points for t in dataset.trajs])
                except Exception:
                    pass
            return fallback_value

    def explain(self):
        try:
            Z_trajs = [
                Trajectory(points=Z_traj) for Z_traj in self._perturbed_traj_generator()
            ]
            labels = [1] * (len(Z_trajs) - 1) + [0]
            Z_pro = Dataset("custom", Z_trajs, labels)

            fallback = np.zeros((len(Z_trajs), 1))
            result = self._safe_predict(Z_pro, fallback)
            
            preds = result
            
            try:
                if isinstance(preds, tuple):
                    pred_labels = [pred.argmax() if hasattr(pred, 'argmax') else pred for pred in preds[0]]
                elif isinstance(preds, np.ndarray) and len(preds.shape) == 2 and preds.shape[1] > 1:
                    # Handle probability outputs from neural networks
                    pred_labels = np.argmax(preds, axis=1)
                else:
                    pred_labels = [pred.argmax() if hasattr(pred, 'argmax') else pred for pred in preds]
                
                # Try to inverse transform if encoder is available
                if hasattr(self.model, 'encoder'):
                    Y = self.model.encoder.inverse_transform(pred_labels)
                elif hasattr(self.model, 'classes_'):
                    # For scikit-learn models
                    Y = [self.model.classes_[label] if isinstance(label, (int, np.integer)) and label < len(self.model.classes_) else label for label in pred_labels]
                else:
                    Y = pred_labels
            except Exception as e:
                if isinstance(preds, tuple):
                    pred_labels = preds[0]
                    Y = pred_labels
                else:
                    Y = preds

            # Ensure Y is converted to a consistent format
            if isinstance(Y, np.ndarray) and Y.ndim == 1:
                Y = Y.tolist()
            elif not isinstance(Y, list):
                Y = [Y]

            if len(np.unique(Y)) == 1:
                return None

            clf = LogisticRegression()
            weights = self._calculate_weight()
            # Convert Y to integers if they're strings for LogisticRegression
            if Y and isinstance(Y[0], str):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                Y_encoded = le.fit_transform(Y)
                clf.fit(self.perturb_vectors, Y_encoded, sample_weight=weights)
            else:
                clf.fit(self.perturb_vectors, Y, sample_weight=weights)

            self.coef_ = clf.coef_
            self.classes_ = clf.classes_
            return self.coef_

        except Exception as e:
            print(f"Error in explain: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((1, self.x_len))

    def get_Y_eval_sorted(self):
        try:
            Z_trajs = [Trajectory(points=Z_traj) for Z_traj in self.Z_eval]
            labels = [1] * (len(Z_trajs) - 1) + [0]
            Z_pro = Dataset("custom1", Z_trajs, labels)

            fallback = np.zeros(len(Z_trajs))
            result = self._safe_predict(Z_pro, fallback)
            
            if isinstance(result, tuple):
                if len(result) >= 2:
                    Y = result[0]
                else:
                    Y = result
            elif isinstance(result, np.ndarray) and len(result.shape) == 2 and result.shape[1] > 1:
                # Handle probability outputs from neural networks
                Y = np.argmax(result, axis=1)
                if hasattr(Y, 'tolist'):
                    Y = Y.tolist()
            else:
                Y = result

            if Y is None:
                return fallback

            try:
                Y_without_pertub = self.get_Y()
            except Exception:
                Y_without_pertub = []

            try:
                class_index = None
                if hasattr(self, 'classes_') and hasattr(self, 'coef_'):
                    if len(Y_without_pertub) > 0:
                        # Try to find the corresponding class index
                        try:
                            target_class = Y_without_pertub[0]
                            if isinstance(target_class, str) and all(isinstance(c, (int, np.integer)) for c in self.classes_):
                                # Convert string to int if needed
                                target_class = int(target_class)
                            class_index = np.where(self.classes_ == target_class)[0][0]
                            class_coef = self.coef_[class_index]
                        except (ValueError, IndexError, TypeError):
                            # If we can't find the class, use the first coefficient vector
                            class_coef = self.coef_[0]
                    else:
                        class_coef = self.coef_[0]
                else:
                    class_coef = np.zeros(len(self.Z_eval))
                
                sorted_indices = np.argsort(abs(class_coef))[::-1]
                
                result = []
                for i in sorted_indices:
                    if isinstance(Y, list) and i < len(Y):
                        result.append(Y[i])
                    elif hasattr(Y, '__getitem__') and i < len(Y):
                        result.append(Y[i])
                
                return result
            except Exception as e:
                print(f"Error in get_Y_eval_sorted processing: {e}")
                import traceback
                traceback.print_exc()
                return Y

        except Exception as e:
            print(f"Error in get_Y_eval_sorted: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(len(self.Z_eval))

    def get_Y(self):
        try:
            Z_trajs = [Trajectory(points=self.X)]
            labels = [0]
            Z_pro = Dataset("custom1", Z_trajs, labels)

            fallback = []
            result = self._safe_predict(Z_pro, fallback)
            
            if isinstance(result, tuple):
                if len(result) >= 2:
                    Y = result[0]
                else:
                    Y = result
            elif isinstance(result, np.ndarray) and len(result.shape) == 2 and result.shape[1] > 1:
                # Handle probability outputs from neural networks
                Y = np.argmax(result, axis=1)
            else:
                Y = result
            
            if not isinstance(Y, list):
                if hasattr(Y, 'tolist'):
                    Y = Y.tolist()
                else:
                    Y = [Y]
                    
            return Y

        except Exception as e:
            print(f"Error in get_Y: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_segment(self):
        try:
            return self.segments
        except Exception as e:
            print(f"Error in get_segment: {e}")
            return []
