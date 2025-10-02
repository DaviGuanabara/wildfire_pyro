from sklearn.preprocessing import StandardScaler
import numpy as np
from wildfire_pyro.environments.iowa.components.adapter_params import AdapterParams


class CustomScaler:
    def __init__(self, adapter_params: AdapterParams):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # deltas são normalizados por constantes
        self.max_delta_time = adapter_params.max_delta_time
        self.max_delta_distance = adapter_params.max_delta_distance

    def fit(self, features: np.ndarray, targets: np.ndarray):
        """
        Fit scalers on the whole dataset.
        features: shape (N, F)
        targets: shape (N, T)
        """
        self.feature_scaler.fit(features)
        self.target_scaler.fit(targets)

    def transform_features(self, neighbor: np.ndarray) -> np.ndarray:
    
        return self.feature_scaler.transform(neighbor)

    def transform_target(self, target: np.ndarray) -> np.ndarray:
        """
        Normalize a single ground truth (T,).
        """
        return self.target_scaler.transform(target.reshape(1, -1)).flatten()
    
    


    def inverse_transform_target(self, target_scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(target_scaled.reshape(1, -1)).flatten()

    def normalize_delta_time(self, delta_time: np.ndarray):
        return delta_time / self.max_delta_time

    def normalize_delta_pos(self, delta_pos: np.ndarray):
        return delta_pos / self.max_delta_distance

    def denormalize_delta_time(self, delta_time_norm: np.ndarray):
        return delta_time_norm * self.max_delta_time

    def denormalize_delta_pos(self, delta_pos_norm: np.ndarray):
        return delta_pos_norm * self.max_delta_distance
