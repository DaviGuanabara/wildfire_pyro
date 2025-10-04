from typing import Any, Dict, Optional
import numpy as np
from wildfire_pyro.environments.base_environment import BaseEnvironment
from wildfire_pyro.environments.iowa.components.meta_data import Metadata
from wildfire_pyro.environments.iowa.components.dataset_adapter import (
    AdapterParams,
    DatasetAdapter,
)
from gymnasium import spaces


class AdaptativeEnvironment(BaseEnvironment):

    def __init__(self, data_path, metadata: Metadata, params: AdapterParams):
        self.dataset_metadata: Metadata = metadata
        self.dataset_adapter = DatasetAdapter(
            data_path=data_path, metadata=metadata, params=params
        )

        self.verbose = params.verbose

        self._compute_spaces()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        
        self.dataset_adapter.reset(self.rng)
        sample, padded, mask, feature_names, ground_truth, terminated = self.dataset_adapter.next()

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

        #print("Step - neighbors:", observation["neighbors"], "mask:", observation["mask"])
        info = {
            "sample": sample,
            "feature_names": feature_names,
            "ground_truth": ground_truth,
        }

        reward = 0
        truncated = False

        return observation, reward, terminated, truncated, info

    def get_bootstrap_observations(
        self, n_bootstrap: int, force_recompute: bool = True
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
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
        if force_recompute:
            self.dataset_adapter.reset(self.rng)

        obs_list = []
        gt_list = []

        for _ in range(n_bootstrap):
            sample, padded, mask, feature_names, ground_truth, terminated = self.dataset_adapter.read_resample_neighbors()

            # 🔹 Cada observação vira um dict compatível com observation_space
            obs_dict = {
                "neighbors": padded.astype(np.float32),
                "mask": mask.astype(np.float32),
            }

            obs_list.append(obs_dict)
            gt_list.append(ground_truth.astype(np.float32))

            if terminated:
                break  # dataset acabou

        ground_truths = np.stack(gt_list, axis=0)

        self.last_observations = obs_list  
        self.last_ground_truths = ground_truths
        return obs_list, ground_truths
    
    def baseline(
        self, observations: list[dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Compute a baseline prediction over the given observations.

        Args:
            observations: list of dicts matching observation_space.
                        Each dict must contain keys 'neighbors' and 'mask'.

        Returns:
            predictions: np.ndarray of shape (n_bootstrap, 1)
        """

        predictions = []

        for obs in observations:
            neighbors = obs["neighbors"]
            mask = obs["mask"].astype(bool)


            valid = neighbors[mask]

            if valid.size == 0 or np.isnan(valid).all():
                predictions.append(np.nan)
            else:
                predictions.append(np.nanmean(valid))

        return np.array(predictions, dtype=np.float32).reshape(-1, 1)
