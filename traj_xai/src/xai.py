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
            print(f"[DEBUG] Attempting standard predict call")
            return self.model.predict(dataset)
        except ValueError as e:
            if "not enough values to unpack" in str(e):
                print(f"[DEBUG] Standard predict failed: {e}, trying alternative approach")
                try:
                    if hasattr(dataset, 'trajs'):
                        print(f"[DEBUG] Trying predict with just trajectories")
                        return self.model.predict(dataset.trajs)
                    elif hasattr(self.model, 'predict_raw'):
                        print(f"[DEBUG] Trying predict_raw method")
                        return self.model.predict_raw(dataset)
                    elif hasattr(self.model, '_get_input_data'):
                        print(f"[DEBUG] Trying with custom input processing")
                        # Try to provide the expected values manually
                        try:
                            x_data, y_data = self.model._get_input_data(dataset)
                            lengths = [len(x) for x in x_data]
                            return self.model._predict_with_data(x_data, lengths)
                        except Exception as inner_e:
                            print(f"[DEBUG] Custom input processing failed: {inner_e}")
                    else:
                        print(f"[DEBUG] Trying direct prediction from trajectories")
                        if isinstance(dataset, Dataset) and hasattr(dataset, 'trajs'):
                            return self.model.predict([t.points for t in dataset.trajs])
                except Exception as alt_e:
                    print(f"[DEBUG] Alternative predict approach failed: {alt_e}")
            print(f"[DEBUG] Predict failed: {e}, returning fallback value")
            return fallback_value

    def explain(self):
        try:
            print(f"[DEBUG] Pre-explain segments: {len(self.segments)}")

            Z_trajs = [
                Trajectory(points=Z_traj) for Z_traj in self._perturbed_traj_generator()
            ]
            labels = [1] * (len(Z_trajs) - 1) + [0]
            Z_pro = Dataset("custom", Z_trajs, labels)

            print("[DEBUG] About to call model.predict() in explain()")
            fallback = np.zeros((len(Z_trajs), 1))
            result = self._safe_predict(Z_pro, fallback)
            
            print(f"[DEBUG] model.predict() result type: {type(result)}")
            if isinstance(result, tuple):
                print(f"[DEBUG] model.predict() returned tuple of length {len(result)}")
                for i, item in enumerate(result):
                    print(f"[DEBUG] Tuple item {i} type: {type(item)}, shape: {getattr(item, 'shape', 'N/A')}")
            elif isinstance(result, np.ndarray):
                print(f"[DEBUG] model.predict() returned ndarray of shape {result.shape}")
            else:
                print(f"[DEBUG] model.predict() value: {result}")

            preds = result
            
            try:
                print(f"[DEBUG] preds type: {type(preds)}, value: {preds}")
                
                if isinstance(preds, tuple):
                    print(f"[DEBUG] Handling tuple prediction with {len(preds)} elements")
                    pred_labels = [pred.argmax() if hasattr(pred, 'argmax') else pred for pred in preds[0]]
                else:
                    pred_labels = [pred.argmax() if hasattr(pred, 'argmax') else pred for pred in preds]
                
                print(f"[DEBUG] pred_labels: {pred_labels}")
                print(f"[DEBUG] encoder type: {type(self.model.encoder)}")
                Y = self.model.encoder.inverse_transform(pred_labels)
                print(f"[DEBUG] Y after inverse_transform: {Y}")
            except Exception as e:
                print(f"[DEBUG] Error in prediction processing: {e}")
                if isinstance(preds, tuple):
                    print(f"[DEBUG] Attempting alternative tuple handling")
                    pred_labels = preds[0]
                    Y = pred_labels
                else:
                    Y = preds
                print(f"[DEBUG] Y using fallback approach: {Y}")

            if len(np.unique(Y)) == 1:
                return None

            clf = LogisticRegression()
            weights = self._calculate_weight()
            print(f"[DEBUG] weights length: {len(weights)}, perturb_vectors length: {len(self.perturb_vectors)}")
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

            print("[DEBUG] About to call model.predict() in get_Y_eval_sorted()")
            fallback = np.zeros(len(Z_trajs))
            result = self._safe_predict(Z_pro, fallback)
            
            print(f"[DEBUG] model.predict() in get_Y_eval_sorted type: {type(result)}")
            if isinstance(result, tuple):
                print(f"[DEBUG] Tuple prediction with {len(result)} elements: {[type(x) for x in result]}")
                if len(result) >= 2:
                    Y = result[0]
                    print(f"[DEBUG] Using first element of tuple as Y: {type(Y)}")
                else:
                    Y = result
            else:
                print(f"[DEBUG] model.predict() value: {result}")
                Y = result

            if Y is None:
                return fallback

            try:
                print("[DEBUG] About to call get_Y()")
                Y_without_pertub = self.get_Y()
                print(f"[DEBUG] Y_without_pertub: {Y_without_pertub}")
            except Exception as e:
                print(f"[DEBUG] Error in get_Y call: {e}")
                Y_without_pertub = []

            try:
                print(f"[DEBUG] self.classes_: {getattr(self, 'classes_', 'not set')}")
                print(f"[DEBUG] self.coef_: {getattr(self, 'coef_', 'not set')}")
                
                class_index = None
                if hasattr(self, 'classes_') and hasattr(self, 'coef_'):
                    if len(np.unique(Y)) > 2 and len(Y_without_pertub) > 0:
                        print(f"[DEBUG] Finding class index for {Y_without_pertub[0]} in {self.classes_}")
                        class_index = np.where(self.classes_ == Y_without_pertub[0])[0][0]
                        class_coef = self.coef_[class_index]
                    else:
                        class_coef = self.coef_[0]
                else:
                    print("[DEBUG] classes_ or coef_ not available, using zeros")
                    class_coef = np.zeros(len(self.Z_eval))
                
                print(f"[DEBUG] class_coef: {class_coef}")
                sorted_indices = np.argsort(abs(class_coef))[::-1]
                print(f"[DEBUG] sorted_indices: {sorted_indices}")
                
                result = []
                for i in sorted_indices:
                    if isinstance(Y, list) and i < len(Y):
                        result.append(Y[i])
                    elif hasattr(Y, '__getitem__') and i < len(Y):
                        result.append(Y[i])
                    else:
                        print(f"[DEBUG] Index {i} out of bounds for Y of type {type(Y)} with length {getattr(Y, '__len__', lambda: 'unknown')()}")
                
                return result
            except Exception as e:
                print(f"[DEBUG] Error in sorting/processing: {e}")
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

            print("[DEBUG] About to call model.predict() in get_Y()")
            fallback = []
            result = self._safe_predict(Z_pro, fallback)
            
            print(f"[DEBUG] model.predict() in get_Y type: {type(result)}")
            if isinstance(result, tuple):
                print(f"[DEBUG] Tuple prediction in get_Y with {len(result)} elements")
                for i, item in enumerate(result):
                    print(f"[DEBUG] Tuple item {i} type: {type(item)}, shape: {getattr(item, 'shape', 'N/A')}")
                if len(result) >= 2:
                    Y = result[0]
                else:
                    Y = result
            else:
                print(f"[DEBUG] model.predict() value: {result}")
                Y = result
            
            if not isinstance(Y, list):
                if hasattr(Y, 'tolist'):
                    Y = Y.tolist()
                else:
                    Y = [Y]
                    
            print(f"[DEBUG] Returning Y: {Y}")
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
