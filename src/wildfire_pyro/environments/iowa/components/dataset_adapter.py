import os
import random
from typing import Hashable, List, Optional, Tuple, Any, cast
import pandas as pd
import numpy as np
import logging

from wildfire_pyro.environments.iowa.components.adapter_params import AdapterParams
from wildfire_pyro.environments.iowa.components.custom_scale import CustomScaler
from wildfire_pyro.environments.iowa.components.meta_data import Metadata

logger = logging.getLogger("SensorManager")
logger.setLevel(logging.INFO)



class DatasetAdapter:

    def __init__(self, data_path, metadata: Metadata, params: AdapterParams, rng: np.random.Generator = np.random.default_rng(),
                 scaler: Optional[CustomScaler] = None):
        self.data_path = data_path
        self.metadata = metadata
        self.params = params

        self.scaler = scaler or CustomScaler(self.params)

        self.load_data(self.data_path, metadata)

        self.reset(rng)

    def reset(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self._set_shapes()

        self.cursor = self.params.min_neighborhood_size
        self.done = False

    def _set_shapes(self):
        self.neighbors_shape, self.mask_shape, self.ground_truth_shape = (
            self._get_shapes()
        )

    def _load_data(self, data_path: str, metadata: Optional[Metadata]) -> pd.DataFrame:
        self.data_path = data_path
        ext = os.path.splitext(data_path)[1].lower()

        if ext == ".csv":
            data = pd.read_csv(data_path)
        elif ext in [".xls", ".xlsx"]:
            data = pd.read_excel(data_path)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {ext}")

        if metadata is not None:
            self.metadata = metadata
        elif not hasattr(self, "metadata"):
            raise ValueError("Metadata must be provided on first load.")

        self.validate(data, self.metadata)
        self.data = self.sort_by_time(self.metadata, data)
        return self.data

    def load_data(
        self, data_path: str, metadata: Optional[Metadata] = None
    ) -> pd.DataFrame:
        df = self._load_data(data_path, metadata)
        if self.params.verbose:
            logger.info(f"Data loaded successfully from {data_path}")
            df.info()
            logger.info(f"Metadata: {self.metadata}")

        # split features and targets
        feature_cols = [c for c in df.columns
                        if c not in ([self.metadata.id]
                                    + (self.metadata.exclude or []))]

        features = df[feature_cols].to_numpy(dtype=float)
        targets = df[self.metadata.target].to_numpy(dtype=float)


        self.scaler.fit(features, targets)
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
    def filter_by_id(self, candidates: pd.DataFrame, row) -> pd.DataFrame:
        id_col = self.metadata.id
        return candidates[candidates[id_col] != row[id_col]]

    def filter_by_time(
        self, candidates: pd.DataFrame, row, delta_time: Optional[float]
    ) -> pd.DataFrame:
        if delta_time is None:
            return candidates
        time_col = self.metadata.time
        dt = np.abs(candidates[time_col] - row[time_col])
        return candidates[dt <= delta_time]

    def filter_by_distance(
        self, candidates: pd.DataFrame, row, distance: Optional[float]
    ) -> pd.DataFrame:
        if distance is None:
            return candidates
        pos_cols = self.metadata.position
        ref = row[pos_cols].values.astype(float)
        coords = candidates[pos_cols].values.astype(float)
        dists = np.linalg.norm(coords - ref, axis=1)
        return candidates[dists <= distance]

    def filter_by_index(
        self, candidates: pd.DataFrame, row_index: Optional[int]
    ) -> pd.DataFrame:
        if row_index is None:
            return candidates
        return candidates.loc[candidates.index < row_index]

    def random_choice(self, candidates):
        n_neighbors = self.rng.integers(
            low=self.params.min_neighborhood_size,
            high=self.params.max_neighborhood_size + 1
        )
        k = min(n_neighbors, len(candidates))

        if k == 0:
            return candidates.iloc[[]]  # empty DataFrame with same columns

        idx = self.rng.choice(candidates.index, size=k, replace=False)
        return candidates.loc[idx]

    def get_neighbors(
        self,
        row_index: int,
        row: pd.Series,
        ) -> pd.DataFrame:

        candidates = self.data
        candidates = self.filter_by_index(candidates, row_index)
        candidates = self.filter_by_id(candidates, row)
        candidates = self.filter_by_time(candidates, row, self.params.max_delta_time)
        candidates = self.filter_by_distance(candidates, row, self.params.max_delta_distance)

        return self.random_choice(candidates)


    def _compute_deltas(
        self, row: pd.Series, neighbors: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        time_col = self.metadata.time
        pos_cols = self.metadata.position

        # Δ tempo: vetor (num_neighbors,)
        delta_time = neighbors[time_col].to_numpy() - row[time_col]

        # Δ posição: matriz (num_neighbors, num_pos_cols)
        ref_pos = row[pos_cols].to_numpy(dtype=float)
        coords = neighbors[pos_cols].to_numpy(dtype=float)
        delta_pos = coords - ref_pos  # dif por coordenada

        return delta_time, delta_pos

    def _add_deltas(
        self,
        formatted: pd.DataFrame,
        row: pd.Series,
        neighbors: pd.DataFrame,
        max_delta_distance: float,
        max_delta_time: float,
    ) -> pd.DataFrame:
        delta_time, delta_pos = self._compute_deltas(row, neighbors)
        delta_time = delta_time / max_delta_time
        delta_pos = delta_pos / max_delta_distance

        formatted["delta_time"] = delta_time
        for i, col in enumerate(self.metadata.position):
            formatted[f"delta_{col}"] = delta_pos[:, i]

        return formatted

    def _add_targets(
        self, formatted: pd.DataFrame, neighbors: pd.DataFrame
    ) -> pd.DataFrame:
        for tgt in self.metadata.target:
            if tgt in neighbors.columns:
                formatted[f"target_{tgt}"] = neighbors[tgt].values
        return formatted

    def _add_features(
        self, formatted: pd.DataFrame, neighbors: pd.DataFrame
    ) -> pd.DataFrame:
        exclude_cols = {
            self.metadata.id,
            self.metadata.time,
            *self.metadata.position,
            *self.metadata.target,
            *(self.metadata.exclude or []),  # 👈 inclui exclude aqui
        }
        feature_cols = [c for c in neighbors.columns if c not in exclude_cols]
        for col in feature_cols:
            formatted[f"feat_{col}"] = neighbors[col].values
        return formatted

    def _get_shapes(self) -> Tuple[Tuple[int, int], Tuple[int], Tuple[int]]:
        """
        Return the shapes of (padded, mask, ground_truth).
        Useful for defining the observation_space in Gymnasium.
        """

        # Mask shape: always (M,)
        mask_shape = (self.params.max_neighborhood_size,)

        # Ground truth shape: number of targets
        ground_truth_shape = (len(self.metadata.target),)

        # ⚡ Take a single row to infer the number of features
        sample = self.data.sample(n=1).iloc[0]
        neighbors = self.get_neighbors(
            row_index=cast(int, sample.name),
            row=sample,
        )
        formatted = self.format_neighbors(
            sample,
            neighbors,
            max_delta_distance=self.params.max_delta_distance,
            max_delta_time=self.params.max_delta_time,
        )

        num_features = formatted.shape[1]

        # Padded has shape (M, F)
        padded_shape = (self.params.max_neighborhood_size, num_features)

        return padded_shape, mask_shape, ground_truth_shape



    def format_neighbors(
        self,
        row: pd.Series,
        neighbors: pd.DataFrame,
        max_delta_distance: float,
        max_delta_time: float,
    ) -> pd.DataFrame:

        formatted = pd.DataFrame(index=neighbors.index)

        formatted = self._add_deltas(
            formatted, row, neighbors, max_delta_distance, max_delta_time
        )
        formatted = self._add_targets(formatted, neighbors)
        formatted = self._add_features(formatted, neighbors)

        return formatted

    def pad_neighbors(
        self,
        neighbors: pd.DataFrame,
        max_neighborhood_size: int,
        shuffle: bool = True,
        invalid_value: float = 0.0,
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
            idx = self.rng.permutation(M)
            padded = padded[idx]
            mask = mask[idx]

        return padded, mask

    def get_ground_truth(self, row: pd.Series) -> np.ndarray:
        """Extrai o target (ground truth) do sample central."""
        return row[self.metadata.target].to_numpy(dtype=float)
    
    def normalize_observation(self, padded, ground_truth):
        # 🔹 Normalize features and targets
        padded_scaled = self.scaler.transform_features(padded)
        ground_truth_scaled  = self.scaler.transform_target(ground_truth)
        return padded_scaled, ground_truth_scaled

    def next(self) -> Tuple[pd.Series, np.ndarray, np.ndarray, List[str], np.ndarray, bool]:
        """
        Return next row with its neighborhood (padded).
        Iterates sequentially and flags `done=True` when the dataset ends.
        Normalizes features and targets using the configured scaler.
        """

        if self.cursor >= len(self.data):
            raise Exception("[dataset_adapter] no more data.")

        self.done = False
        sample = self.data.iloc[self.cursor]
        row_index = sample.name
        self.cursor += 1   


        neighbors = self.get_neighbors(
            row_index=cast(int, row_index),
            row=sample
        )

        formatted = self.format_neighbors(
            sample,
            neighbors,
            max_delta_distance=self.params.max_delta_distance,
            max_delta_time=self.params.max_delta_time,
        )

        padded, mask = self.pad_neighbors(
            formatted, max_neighborhood_size=self.params.max_neighborhood_size
        )

        feature_names = list(formatted.columns)
        ground_truth = self.get_ground_truth(sample)

        # 🔹 Normalize features and targets
        padded_scaled, ground_truth_scaled = self.normalize_observation(
            padded, ground_truth)

        return sample, padded_scaled, mask, feature_names, ground_truth_scaled, self.done



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ⚠️ Preencha com o caminho real do seu CSV
    # data_path = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\wildfire_pyro\\examples\\iowa_soil\\data\\ISU_Soil_Moisture_Network\\dataset_preprocessed.xlsx"
    data_path = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\wildfire_pyro\\examples\\iowa_soil\\data\\train.csv"

    # Exemplo de metadados
    metadata = Metadata(
        time="valid",  # coluna de tempo
        position=["Latitude1", "Longitude1", "Elevation [m]"],  # colunas espaciais
        id="station",  # coluna de identificação
        exclude=[
            "out_lwmv_1",
            "out_lwmv_2",
            "out_lwmdry_1_tot",
            "out_lwmcon_1_tot",
            "out_lwmdry_2_tot",
            "out_lwmcon_2_tot",
            "out_lwmwet_2_tot",  # colunas a excluir
            "ID",
            "Archive Begins",
            "Archive Ends",
            "IEM Network",
            "Attributes",
            "Station Name",
        ],
        target=["out_lwmwet_1_tot"],  # , "out_lwmwet_2_tot"]  # colunas alvo
    )

    params = AdapterParams(
        min_neighborhood_size=5,
        max_neighborhood_size=10,
        max_delta_distance=1e9,
        max_delta_time=10.0,
        verbose=True,
    )

    adapter = DatasetAdapter(data_path, metadata, params=params)

    # Lê uma amostra com vizinhança
    sample, padded, mask, feature_names, ground_truth, done = adapter.next()

    print("\n=== Sample (row) ===")
    print(sample)
    
    print("\n=== Padded neighbors ===")
    print(feature_names)
    print(padded)

    print("\n=== Mask ===")
    print(mask)

    print("\n=== Ground Truth ===")
    print(ground_truth)
