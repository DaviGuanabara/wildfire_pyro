import numpy as np
from wildfire_pyro.environments.base_environment import BaseEnvironment
from wildfire_pyro.environments.iowa.components.meta_data import Metadata
from wildfire_pyro.environments.iowa.components.dataset_adapter import (
    AdapterParams,
    DatasetAdapter,
)
from gymnasium import spaces


class SensorEnvironment(BaseEnvironment):

    def __init__(self, data_path, metadata: Metadata, verbose: bool = False):

        params = AdapterParams(
            neighborhood_size=5,
            max_neighborhood_size=10,
            max_delta_distance=1e9,
            max_delta_time=10.0,
            verbose=True,
        )
        self.dataset_adapter = DatasetAdapter(
            data_path=data_path, metadata=metadata, params=params
        )

        self.dataset_adapter.read()

    def _compute_observation_space(self):
        """

        - neighbors: padded neighborhood
        - mask: binary mask
        - ground_truth: target. value varies between 0 up to 1440. So, normalized is 0 to 1.
        """

        padded_shape = self.dataset_adapter.neighbors_shape
        mask_shape = self.dataset_adapter.mask_shape
        ground_truth_shape = self.dataset_adapter.ground_truth_shape

        return spaces.Dict(
            {
                "neighbors": spaces.Box(
                    low=-1, high=1, shape=padded_shape, dtype=np.float32
                ),
                "mask": spaces.Box(low=0, high=1, shape=mask_shape, dtype=np.int8),
                "ground_truth": spaces.Box(
                    low=0, high=1, shape=ground_truth_shape, dtype=np.float32
                ),
            }
        )
