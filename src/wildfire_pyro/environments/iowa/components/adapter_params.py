from dataclasses import dataclass


@dataclass
class AdapterParams:
    min_neighborhood_size: int
    max_neighborhood_size: int
    max_delta_distance: float
    max_delta_time: float
    random_cursor_reposition: bool
    verbose: bool
