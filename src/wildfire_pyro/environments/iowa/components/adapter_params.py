from dataclasses import dataclass


@dataclass
class AdapterParams:
    min_neighborhood_size: int
    max_neighborhood_size: int
    max_delta_distance: float
    max_delta_time: float
    pivot_next_random: bool
    verbose: bool
