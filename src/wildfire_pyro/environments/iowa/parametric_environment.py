from typing import Any, Dict, Optional
import numpy as np
from wildfire_pyro.environments.base_environment import BaseEnvironment
from wildfire_pyro.environments.iowa.components.custom_scale import CustomScaler
from wildfire_pyro.environments.iowa.components.metadata import Metadata
from wildfire_pyro.environments.iowa.components.dataset_adapter import (
    AdapterParams,
    DatasetAdapter,
)
from gymnasium import spaces


from wildfire_pyro.common.baselines.BaselineFactory import BaselineFactory



class ParametricEnvironment(BaseEnvironment):

    def __init__(self, data_path, metadata: Metadata, params: AdapterParams, baseline_type: str = "mean_neighbor",
                 scaler: Optional[CustomScaler] = None):
        self.dataset_metadata: Metadata = metadata
        self.dataset_adapter = DatasetAdapter(
            data_path=data_path, metadata=metadata, params=params, scaler=scaler
        )

        self.verbose = params.verbose

        self._compute_spaces()

        self.baseline_model = BaselineFactory.create_baseline(
            baseline_type=baseline_type,
            observation_space=self.observation_space,
            action_space=self.action_space,
            scaler=self.dataset_adapter.scaler
        )

        self.baseline_type = baseline_type

    def get_fitted_scaler(self) -> CustomScaler:
        return self.dataset_adapter.scaler

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        
        self.dataset_adapter.reset(self.rng)
        sample, padded, mask, feature_names, ground_truth, terminated = self.dataset_adapter.next()
        self.neighbor_schema = self.dataset_adapter.neighbor_schema

        self.neighbor_schema = self.dataset_adapter.neighbor_schema
        self.baseline_model.set_schema(self.neighbor_schema)


        observation = {
            "neighbors": padded.astype(np.float32),
            "mask": mask.astype(np.int8),
        }

        info = {
            "sample": sample,
            "feature_names": feature_names,
            "ground_truth": ground_truth,
        }

        return observation, info


    def _compute_observation_space(self):

        padded_shape = self.dataset_adapter.neighbors_shape
        mask_shape = self.dataset_adapter.mask_shape
        

        observation_space = spaces.Dict(
            {
                "neighbors": spaces.Box(low=-1, high=1, shape=padded_shape, dtype=np.float32),
                "mask": spaces.Box(low=0, high=1, shape=mask_shape, dtype=np.int8),
            }
        )

        return observation_space

    def _compute_action_space(self):
        ground_truth_shape = self.dataset_adapter.ground_truth_shape
        action_space = spaces.Box(
            low=-1, high=1, shape=ground_truth_shape, dtype=np.float32
        )

        return action_space

    def _compute_spaces(self):

        self.observation_space = self._compute_observation_space()
        self.action_space = self._compute_action_space()


    

    def step(self, action: Optional[Any] = None) -> tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        
        sample, padded, mask, feature_names, ground_truth, terminated = self.dataset_adapter.next()
    
        if terminated and self.verbose == True:
            print("Episode terminated.")
            


        observation = {
            "neighbors": padded.astype(np.float32),
            "mask": mask.astype(np.int8),
        }

        info = {
            "sample": sample,
            "feature_names": feature_names,
            "ground_truth": ground_truth,
        }

        reward = 0
        truncated = False

        return observation, reward, terminated, truncated, info
    
    def get_baseline(self):
        """
        Return the baseline values for the current sample.

        Returns:
            np.ndarray: baseline values
        """
        baseline_values = self.dataset_adapter.get_baseline()
        return baseline_values.astype(np.float32)
    

    def get_bootstrap_observations(
        self, n_bootstrap: int, force_recompute: bool = True
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Return a batch of bootstrap observations and their ground truth values.

        Args:
            n_bootstrap (int): number of bootstrap samples.
            force_recompute (bool): if True, restart dataset cursor from beginning.

        Returns:
            tuple:
                - observations: list of dicts with keys matching observation_space
                - ground_truths: array (n_bootstrap, T) with targets
        """
        obs_neighbors = []
        obs_masks = []
        gt_list = []
        baseline_list = []

        for _ in range(n_bootstrap):
            sample, padded, mask, feature_names, ground_truth, terminated = (
                self.dataset_adapter.read_resample_neighbors()
            )

            obs_neighbors.append(padded.astype(np.float32))
            obs_masks.append(mask.astype(np.float32))
            gt_list.append(ground_truth.astype(np.float32))
            baseline_list.append(
                self.dataset_adapter.get_baseline().astype(np.float32))

            if terminated:
                break

        observations = {
            "neighbors": np.stack(obs_neighbors, axis=0),
            "mask": np.stack(obs_masks, axis=0),
        }

        ground_truths = np.stack(gt_list, axis=0)
        bootstrap_baseline = np.stack(baseline_list, axis=0)

        return observations, ground_truths, bootstrap_baseline

    def to_raw_target(self, y: np.ndarray) -> np.ndarray:
        return self.dataset_adapter.scaler.inverse_transform_target(y)
