from dataclasses import dataclass, field
from typing import List, Dict
import pandas as pd


@dataclass
class NeighborSchema:
    target_cols: List[str]
    feature_cols: List[str]
    delta_cols: List[str]
    all_cols: List[str]
    index_map: Dict[str, int]

    @staticmethod
    def from_formatted(
        formatted: pd.DataFrame,
        target_cols: List[str],
        feature_cols: List[str],
        delta_cols: List[str]
    ) -> "NeighborSchema":

        all_cols = list(formatted.columns)
        index_map = {col: idx for idx, col in enumerate(all_cols)}

        return NeighborSchema(
            target_cols=target_cols,
            feature_cols=feature_cols,
            delta_cols=delta_cols,
            all_cols=all_cols,
            index_map=index_map,
        )
