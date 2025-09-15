import random
from typing import Hashable, List, Optional, Tuple, Any, cast
import pandas as pd
import numpy as np
import logging

from wildfire_pyro.environments.iowa.components.meta_data import Metadata

logger = logging.getLogger("SensorManager")
logger.setLevel(logging.INFO)


class DatasetAdapter:

    def __init__(self, data_path, metadata: Metadata, verbose: bool = False):
        self.data_path = data_path
        self.verbose = verbose
        self.load_data(self.data_path, metadata)

    def _load_data(self, data_path: str, metadata: Optional[Metadata]) -> pd.DataFrame:
        self.data_path = data_path
        data = pd.read_excel(data_path)

        if metadata is not None:
            self.metadata = metadata
        elif not hasattr(self, "metadata"):
            raise ValueError("Metadata must be provided on first load.")

        self.validate(data, self.metadata)
        self.data = self.sort_by_time(self.metadata, data)
        return self.data

    def load_data(self, data_path: str, metadata: Optional[Metadata] = None) -> pd.DataFrame:
        df = self._load_data(data_path, metadata)
        if self.verbose:
            logger.info(f"Data loaded successfully from {data_path}")
        return df

    def sort_by_time(self, metadata: Metadata, data: pd.DataFrame):
        return data.sort_values(by=metadata.time).reset_index(drop=True)

    def validate(self, dataframe, metadata: Metadata) -> None:
        missing = []
        for group in [metadata.time, metadata.position, metadata.target]:
            if isinstance(group, list):
                for col in group:
                    if col not in dataframe.columns:
                        missing.append(col)
            else:  # caso metadata.time seja str
                if group not in dataframe.columns:
                    missing.append(group)
        if metadata.id not in dataframe.columns:
            missing.append(metadata.id)
        if missing:
            raise ValueError(f"Missing columns in dataframe: {missing}")

    # -------- Filters -------- #
    def filter_by_id(self, candidates: pd.DataFrame, row, itself_as_neighbor: bool) -> pd.DataFrame:
        id_col = self.metadata.id
        if not itself_as_neighbor:
            return candidates[candidates[id_col] != row[id_col]]
        return candidates

    def filter_by_time(self, candidates: pd.DataFrame, row, delta_time: Optional[float]) -> pd.DataFrame:
        if delta_time is None:
            return candidates
        time_col = self.metadata.time
        dt = np.abs(candidates[time_col] - row[time_col])
        return candidates[dt <= delta_time]

    def filter_by_distance(self, candidates: pd.DataFrame, row, distance: Optional[float]) -> pd.DataFrame:
        if distance is None:
            return candidates
        pos_cols = self.metadata.position
        ref = row[pos_cols].values.astype(float)
        coords = candidates[pos_cols].values.astype(float)
        dists = np.linalg.norm(coords - ref, axis=1)
        return candidates[dists <= distance]

    def filter_by_index(self, candidates: pd.DataFrame, row_index: Optional[int]) -> pd.DataFrame:
        if row_index is None:
            return candidates
        return candidates.loc[candidates.index < row_index]

    # -------- Core logic -------- #
    def get_neighbors(
        self,
        row_index: int,
        row: pd.Series,
        neighborhood_size: int,
        max_delta_distance: Optional[float] = None,
        max_delta_time: Optional[float] = None,
        itself_as_neighbor: bool = False
    ) -> pd.DataFrame:

        candidates = self.data
        candidates = self.filter_by_index(candidates, row_index)
        candidates = self.filter_by_id(candidates, row, itself_as_neighbor)
        candidates = self.filter_by_time(candidates, row, max_delta_time)
        candidates = self.filter_by_distance(candidates, row, max_delta_distance)

        return candidates.sample(n=min(neighborhood_size, len(candidates)))

    def format_neighbors(
        self,
        row: pd.Series,
        neighbors: pd.DataFrame,
        max_delta_distance: float,
        max_delta_time: float
    ) -> pd.DataFrame:
        time_col = self.metadata.time
        pos_cols = self.metadata.position

        # --- metadados calculados ---
        delta_time = (neighbors[time_col].values - row[time_col]) / max_delta_time
        ref_pos = row[pos_cols].values.astype(float)
        coords = neighbors[pos_cols].values.astype(float)
        delta_distance = np.linalg.norm(
            coords - ref_pos, axis=1) / max_delta_distance

        formatted = pd.DataFrame({
            "delta_time": delta_time,
            "delta_distance": delta_distance
        }, index=neighbors.index)

        # --- targets ---
        for tgt in self.metadata.target:
            if tgt in neighbors.columns:
                formatted[tgt] = neighbors[tgt].values

        # --- features adicionais ---
        exclude_cols = {self.metadata.id, self.metadata.time,
                        *self.metadata.position, *self.metadata.target}
        feature_cols = [c for c in neighbors.columns if c not in exclude_cols]
        for col in feature_cols:
            formatted[col] = neighbors[col].values

        return formatted


    def pad_neighbors(
        self,
        neighbors: pd.DataFrame,
        max_neighborhood_size: int,
        shuffle: bool = True,
        invalid_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        k, F = neighbors.shape
        M = max_neighborhood_size


        padded = np.full((M, F), invalid_value, dtype=float)
        mask = np.zeros((M,), dtype=np.bool_)


        use_k = min(k, M)
        if use_k > 0:
            arr = neighbors.values.astype(float)
            padded[:use_k, :] = arr[:use_k]
            mask[:use_k] = True


        if shuffle and M > 1:
            idx = np.arange(M)
            np.random.shuffle(idx)
            padded = padded[idx]
            mask = mask[idx]

        return padded, mask


    def read(
        self,
        neighborhood_size: int,
        max_neighborhood_size: int,
        max_delta_distance: float,
        max_delta_time: float,
        itself_as_neighbor: bool = False
    ) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
        """
        Sample ONE row with its neighborhood (padded).
        """
        sample = self.data.sample(n=1).iloc[0]
        row_index = sample.name

        neighbors = self.get_neighbors(
            row_index=cast(int, row_index),
            row=sample,
            neighborhood_size=neighborhood_size,
            max_delta_distance=max_delta_distance,
            max_delta_time=max_delta_time,
            itself_as_neighbor=itself_as_neighbor
        )

        formatted = self.format_neighbors(
            sample, neighbors,
            max_delta_distance=max_delta_distance,
            max_delta_time=max_delta_time
        )

        padded, mask = self.pad_neighbors(
            formatted, max_neighborhood_size=max_neighborhood_size)

        return sample, padded, mask


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ⚠️ Preencha com o caminho real do seu CSV
    data_path = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\wildfire_pyro\\examples\\iowa_soil\\data\\ISU_Soil_Moisture_Network\\dataset_preprocessed.xlsx"

    # Exemplo de metadados
    metadata = Metadata(
        time="data",  # coluna de tempo
        position=["Latitude1", "Longitude1",
                  "Elevation [m]"],  # colunas espaciais
        id="ID",  # coluna de identificação
        target=["high", "low"]  # colunas alvo
    )

    adapter = DatasetAdapter(data_path, metadata, verbose=True)

    # Lê uma amostra com vizinhança
    sample, padded, mask = adapter.read(
        neighborhood_size=random.randint(1, 5),
        max_neighborhood_size=5,
        max_delta_distance=1e9,
        max_delta_time=10.0,
        itself_as_neighbor=False
    )

    print("\n=== Sample (row) ===")
    print(sample)

    print("\n=== Padded neighbors ===")
    print(padded)

    print("\n=== Mask ===")
    print(mask)
