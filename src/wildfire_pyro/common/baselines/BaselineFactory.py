from wildfire_pyro.common.baselines.BaselineRegistry import BaselineRegistry
from wildfire_pyro.common.baselines.BaselineStrategy import BaselineStrategy
from typing import Type

## IMPORTANT: Ensure baseline implementations are imported so they register themselves
import wildfire_pyro.common.baselines.MeanNeighborBaseline



class BaselineFactory:
    @staticmethod
    def create_baseline(baseline_type, observation_space, action_space, scaler=None) -> BaselineStrategy:
        baseline_class: Type[BaselineStrategy] = BaselineRegistry.get(baseline_type)
        return baseline_class(observation_space, action_space, scaler)
