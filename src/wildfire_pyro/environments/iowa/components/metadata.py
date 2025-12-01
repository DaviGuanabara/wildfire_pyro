from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Metadata:
    time: str                 # "data"
    position: List[str]       # ["Latitude1", "Longitude1"] - deltas
    id: str                   # "ID" (or other name)
    target: List[str]         # ["high", "low"] or other list
    features: List[str]       # features column
