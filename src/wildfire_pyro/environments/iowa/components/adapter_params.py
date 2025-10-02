from dataclasses import dataclass


@dataclass
class AdapterParams:
    min_neighborhood_size: int
    max_neighborhood_size: int
    max_delta_distance: float = 1e9
    max_delta_time: float = 10.0
    verbose: bool = False
