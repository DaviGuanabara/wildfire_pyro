from dataclasses import dataclass
import os
import random
from typing import Hashable, List, Optional, Tuple, Any, cast
import pandas as pd
import numpy as np
import logging

from wildfire_pyro.environments.iowa.components.adapter_params import AdapterParams
from wildfire_pyro.environments.iowa.components.custom_scale import CustomScaler
from wildfire_pyro.environments.iowa.components.metadata import Metadata
from wildfire_pyro.environments.iowa.components.neighbor_schema import NeighborSchema

logger = logging.getLogger("SensorManager")
logger.setLevel(logging.INFO)


# TODO
"""
Refactoring Note: DatasetAdapter Decomposition

This adapter currently combines multiple concerns:
1) dataset I/O and preprocessing,
2) pivot lifecycle and consistency,
3) neighborhood sampling and feature assembly.

To improve correctness, testability, and maintainability, the adapter can be
decomposed into focused components while preserving the current public API.

Recommended architecture
------------------------
- DatasetRepository
  Owns dataset state and metadata:
  - load/clean/validate/sort data
  - manage unique_times and scaler
  - provide indexed access helpers (time slice, row lookup, index mapping)

- PivotManager
  Owns pivot state and rules:
  - generate random/sequential pivots
  - validate pivot consistency across (unique_time_idx, in_time_idx, global_idx)
  - advance pivot for next() according to strategy
  - compute termination flag by current policy

- NeighborhoodBuilder
  Owns context construction:
  - apply candidate filters (id, time, distance, index)
  - sample neighbors with controlled randomness
  - assemble formatted neighborhood (targets, features, deltas)
  - pad and mask neighborhood tensors

- DatasetAdapter (facade)
  Orchestrates components and keeps compatibility with callers:
  - reset()
  - next()
  - read_resample_neighbors()
  - get_baseline()

Behavioral contract to preserve
-------------------------------
- next(): reads using current pivot, then advances pivot.
- read_resample_neighbors(): keeps the same pivot and resamples only neighbors.
- bootstrap usage: target (pivot row) remains fixed, contextual neighbors vary.
- reproducibility: all randomness comes from a seeded RNG path.

This decomposition separates stateful pivot logic from data and feature logic,
reducing regression risk and enabling direct unit tests per responsibility.
"""


@dataclass
class Pivot:
    unique_time_idx: int
    in_time_idx: int
    global_idx: int


class DatasetAdapter:
    "A component that maps a dataset to framework’s internal semantic model."

    def __init__(
        self,
        data_path,
        metadata: Metadata,
        params: AdapterParams,
        seed: int,
        scaler: Optional[CustomScaler] = None,
    ):
        self.data_path = data_path
        self.metadata = metadata
        self.params = params

        self.load_data(self.data_path, scaler=scaler, metadata=metadata)
        self.reset(seed)

    def reset(self, seed: Optional[int] = None):
        """
        Reset the adapter's internal state, including the random seed for reproducibility.
        RNG is created only when a new seed is passed.
        """

        if seed is None and not hasattr(self, "rng"):
            raise ValueError(
                "[DATASET ADAPTER] RNG not initialized; call reset(seed) first."
            )

        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self._set_shapes()

        self.done = False
        self.unique_times = np.sort(self.data[self.metadata.time].unique())

        if self.params.pivot_next_random:
            self.pivot = self._gen_random_pivot()
        else:
            self.pivot = self._gen_sequential_pivot()

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

            if metadata.baseline is not None:
                if len(metadata.baseline) != len(metadata.target):
                    raise ValueError(
                        "baseline and target must have the same dimensionality "
                        f"(got {len(metadata.baseline)} vs {len(metadata.target)})"
                    )

        elif not hasattr(self, "metadata"):
            raise ValueError("Metadata must be provided on first load.")

        self.validate(data, self.metadata)

        data["valid"] = pd.to_datetime(data["valid"]).map(pd.Timestamp.toordinal)

        self.data = self.sort_by_time(self.metadata, data)
        return self.data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Substitui inf e NaN por 0.0 em todas as colunas numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = (
            data[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        return data

    def _split_features_targets(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        feature_cols = self.metadata.features
        # self.feature_cols = feature_cols
        features = df[feature_cols].to_numpy(dtype=float)
        targets = df[self.metadata.target].to_numpy(dtype=float)

        return features, targets

    def load_data(
        self,
        data_path: str,
        scaler: Optional[CustomScaler] = None,
        metadata: Optional[Metadata] = None,
    ) -> pd.DataFrame:
        df = self._load_data(data_path, metadata)

        df = self._clean_data(df)

        if self.params.verbose:
            logger.info(f"Data loaded successfully from {data_path}")
            df.info()
            logger.info(f"Metadata: {self.metadata}")

        # split features and targets
        features, targets = self._split_features_targets(df)

        if scaler is None:
            scaler = CustomScaler(self.params)
            scaler.fit(features, targets)

        self.scaler: CustomScaler = scaler
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

        if not hasattr(self, "rng"):
            raise ValueError(
                "Random number generator not initialized. Call reset(seed) before using this method."
            )

        n_neighbors = self.rng.integers(
            low=self.params.min_neighborhood_size,
            high=self.params.max_neighborhood_size + 1,
        )
        k = min(n_neighbors, len(candidates))

        if k == 0:
            return candidates.iloc[[]]  # empty DataFrame with same columns

        idx = self.rng.choice(candidates.index, size=k, replace=True)
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
        candidates = self.filter_by_distance(
            candidates, row, self.params.max_delta_distance
        )

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
        self, formatted: pd.DataFrame, row: pd.Series, neighbors: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        delta_time, delta_pos = self._compute_deltas(row, neighbors)

        # Normalize using the CustomScaler — not hardcoded logic here
        if self.scaler is None:
            raise ValueError("Scaler must be initialized before normalizing deltas.")

        delta_time_norm = self.scaler.normalize_delta_time(delta_time)
        delta_pos_norm = self.scaler.normalize_delta_pos(delta_pos)

        formatted["delta_time"] = delta_time_norm

        # delta_x, delta_y, delta_z ... depending on metadata.position
        new_cols = ["delta_time"]
        for i, col in enumerate(self.metadata.position):
            formatted[f"delta_{col}"] = delta_pos_norm[:, i]
            new_cols.append(f"delta_{col}")

        return formatted, new_cols

    def _add_targets(
        self, formatted: pd.DataFrame, neighbors: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:

        if self.scaler is None:
            raise ValueError("Scaler must be initialized before adding targets.")

        raw = neighbors[self.metadata.target].to_numpy(dtype=float)
        scaled = self.scaler.transform_target(raw)
        new_cols = []
        for i, tgt in enumerate(self.metadata.target):
            formatted[f"{tgt}"] = scaled[:, i]
            new_cols.append(f"{tgt}")

        return formatted, new_cols

    def _add_features(
        self, formatted: pd.DataFrame, neighbors: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:

        if self.scaler is None:
            raise ValueError("Scaler must be initialized before adding features.")

        # 1) Extract raw feature matrix
        raw = neighbors[self.metadata.features].to_numpy(dtype=float)

        # 2) Scale it
        scaled = self.scaler.transform_features(raw)

        # 3) Insert into formatted
        new_cols = []
        for i, col in enumerate(self.metadata.features):
            formatted[col] = scaled[:, i]
            new_cols.append(col)

        return formatted, new_cols

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
        # sample = self.data.sample(n=1).iloc[0]
        idx = self.rng.integers(len(self.data))
        sample = self.data.iloc[idx]

        neighbors = self.get_neighbors(
            row_index=cast(int, sample.name),
            row=sample,
        )
        formatted = self.format_neighbors(
            sample,
            neighbors,
        )

        num_features = formatted.shape[1]

        # Padded has shape (M, F)
        padded_shape = (self.params.max_neighborhood_size, num_features)

        return padded_shape, mask_shape, ground_truth_shape

    def sort_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        sorted_cols = self.metadata.target + self.metadata.features
        return df[sorted_cols]

    def format_neighbors(
        self,
        row: pd.Series,
        neighbors: pd.DataFrame,
    ) -> pd.DataFrame:

        formatted: pd.DataFrame = pd.DataFrame(index=neighbors.index)
        formatted, target_cols = self._add_targets(formatted, neighbors)
        formatted, feature_cols = self._add_features(formatted, neighbors)
        formatted, delta_cols = self._add_deltas(formatted, row, neighbors)

        if not hasattr(self, "neighbor_schema"):
            self.neighbor_schema = NeighborSchema.from_formatted(
                formatted, target_cols, feature_cols, delta_cols
            )

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

        if not hasattr(self, "rng"):
            raise ValueError(
                "Random number generator not initialized. Call reset(seed) before using this method."
            )

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

        if self.scaler is None:
            raise ValueError(
                "Scaler must be initialized before normalizing observations."
            )

        # 🔹 Normalize features and targets
        padded_scaled = self.scaler.transform_features(padded)
        ground_truth_scaled = self.scaler.transform_target(ground_truth)

        return padded_scaled, ground_truth_scaled

    def _save_last_data(
        self, sample: pd.Series, padded_scaled, mask, feature_names, ground_truth_scaled
    ):
        self.last_sample: pd.Series = sample
        self.last_padded = padded_scaled
        self.last_mask = mask
        self.last_feature_names = feature_names
        self.last_ground_truth = ground_truth_scaled

    def _get_sample(self, pivot: Pivot) -> pd.Series:

        self._validate_pivot(pivot)

        sample = self.data.iloc[pivot.global_idx]
        return sample

    def get_baseline(self) -> np.ndarray:
        """
        Return baseline prediction in the same (scaled) space as the target.

        Returns:
            np.ndarray: baseline values
            Baseline prediction in the same (scaled) space as the target.
        """

        if self.metadata.baseline is None:
            raise ValueError("Metadata baseline columns are not defined.")

        baseline_raw = self.last_sample[self.metadata.baseline].to_numpy(dtype=float)

        baseline_scaled = self.scaler.transform_target(
            baseline_raw.reshape(1, -1)
        ).flatten()

        return baseline_scaled

    def _verify_terminated_sequential(self, pivot: Pivot) -> bool:
        self.done = pivot.global_idx >= (len(self.data) - 1)
        return self.done

    def _verify_terminated_random(self, pivot: Pivot) -> bool:
        if pivot.unique_time_idx >= len(self.unique_times) - 1:
            self.done = True
        else:
            self.done = False
        return self.done

    def _verify_terminated(self, pivot: Pivot) -> bool:
        if self.params.pivot_next_random:
            return self._verify_terminated_random(pivot)
        else:
            return self._verify_terminated_sequential(pivot)

    def _read(
        self, pivot: Pivot
    ) -> Tuple[pd.Series, np.ndarray, np.ndarray, List[str], np.ndarray, bool]:
        """
        Return the row pointed by the pivot, with its neighborhood, formatted and padded.
        """

        if self.scaler is None:
            raise ValueError("Scaler must be initialized before transform targets.")

        done = self._verify_terminated(pivot)

        sample = self._get_sample(pivot)
        row_index = pivot.global_idx
        neighbors = self.get_neighbors(row_index=row_index, row=sample)

        formatted = self.format_neighbors(sample, neighbors)

        feature_names = list(formatted.columns)
        ground_truth = self.get_ground_truth(sample)
        ground_truth_scaled = self.scaler.transform_target(
            ground_truth.reshape(1, -1)
        ).flatten()

        padded, mask = self.pad_neighbors(
            formatted, max_neighborhood_size=self.params.max_neighborhood_size
        )

        # padded_scaled, ground_truth_scaled = self.normalize_observation(padded, ground_truth)
        self._save_last_data(sample, padded, mask, feature_names, ground_truth_scaled)

        return sample, padded, mask, feature_names, ground_truth_scaled, done

    def read_resample_neighbors(self):
        """
        Keep pivot in the same position but resample the neighborhood.
            - Useful for bootstrapping.
            - Note: it also re-shuffles the neighborhood, so the order of neighbors is not guaranteed
            to be the same as in the previous read.
        """
        if not hasattr(self, "rng"):
            raise ValueError(
                "Random number generator not initialized. Call reset(seed) before using this method."
            )
        if not hasattr(self, "pivot") or self.pivot is None:
            raise ValueError("Pivot not initialized. Call next() first.")
        return self._read(self.pivot)

    def next(
        self,
    ) -> Tuple[pd.Series, np.ndarray, np.ndarray, List[str], np.ndarray, bool]:
        """
        Return next pivot with its neighborhood.
        Iterates sequentially if flag pivot_next_random is set as false, and flags `done=True` when the dataset ends.
        """

        if hasattr(self, "done") and self.done:
            raise StopIteration(
                "[DatasetAdapter] No more samples available. Reset the adapter to start over."
            )

        sample, padded, mask, feature_names, ground_truth_scaled, done = self._read(
            self.pivot
        )

        if not done:
            self._advance_pivot()

        return sample, padded, mask, feature_names, ground_truth_scaled, done

    def last(
        self,
    ) -> Tuple[pd.Series, np.ndarray, np.ndarray, List[str], np.ndarray, bool]:
        """Returns the last sample returned by `next()`."""
        return (
            self.last_sample,
            self.last_padded,
            self.last_mask,
            self.last_feature_names,
            self.last_ground_truth,
            self.done,
        )

    # ==================================
    # Pivot
    # ==================================

    def _gen_sequential_pivot(self) -> Pivot:
        init_global_idx = int(self.params.max_delta_time)
        return self._pivot_from_global_idx(init_global_idx)

    def _gen_random_pivot(self) -> Pivot:
        """
        Generate a random pivot that respects the max_delta_time constraint from the dataset start.
        Random day, and then random sensor within the day slice.
        """
        if not hasattr(self, "rng"):
            raise ValueError(
                "Random number generator not initialized. Call reset(seed) before using this method."
            )

        # Choose Unique Time Index
        unique_time_idx = int(
            self.rng.integers(
                int(self.params.max_delta_time),
                len(self.unique_times),
            )
        )

        # Calculate the time slice once
        current_time = self.unique_times[unique_time_idx]
        time_slice = self.data[self.data[self.metadata.time] == current_time]

        # Choose a row within the time slice
        in_time_idx = int(self.rng.integers(0, len(time_slice)))

        # Derive the global index
        global_idx = int(time_slice.index[in_time_idx])

        pivot = Pivot(
            unique_time_idx=unique_time_idx,
            in_time_idx=in_time_idx,
            global_idx=global_idx,
        )

        return pivot

    def _next_sequential_pivot(self, current_pivot: Pivot) -> Pivot:
        """
        Given current pivot, return the next pivot in sequential order.
            - If the next unique time index exceeds the dataset, raises an exception.

        """

        self._validate_pivot(current_pivot)

        next_global_idx = current_pivot.global_idx + 1
        if next_global_idx >= len(self.data):
            raise StopIteration("No more pivots in sequential mode.")

        return self._pivot_from_global_idx(next_global_idx)

    def _pivot_from_global_idx(self, global_idx: int) -> Pivot:
        if global_idx < 0 or global_idx >= len(self.data):
            raise ValueError(f"global_idx out of range: {global_idx}")

        row = self.data.iloc[global_idx]
        row_time = row[self.metadata.time]

        # Find unique_time_idx from the timestamp
        unique_time_idx = int(np.searchsorted(self.unique_times, row_time))
        if (
            unique_time_idx >= len(self.unique_times)
            or self.unique_times[unique_time_idx] != row_time
        ):
            raise ValueError("Row timestamp not found in unique_times.")

        # Build time slice and derive in_time_idx from global position inside that slice
        time_slice = self.data[self.data[self.metadata.time] == row_time]
        positions = np.where(time_slice.index.to_numpy() == global_idx)[0]
        if len(positions) != 1:
            raise ValueError("Could not derive in_time_idx from global_idx.")
        in_time_idx = int(positions[0])

        return Pivot(
            unique_time_idx=unique_time_idx,
            in_time_idx=in_time_idx,
            global_idx=global_idx,
        )

    def _advance_pivot(self):
        if self.params.pivot_next_random:
            self.pivot = self._gen_random_pivot()
        else:
            self.pivot = self._next_sequential_pivot(self.pivot)

    def _validate_pivot(self, pivot: Pivot) -> None:
        if not isinstance(pivot, Pivot):
            raise TypeError(f"Expected Pivot, got {type(pivot)}")

        if not hasattr(self, "unique_times"):
            raise ValueError("unique_times not initialized. Call reset() first.")

        n_times = len(self.unique_times)
        n_rows = len(self.data)

        # 1) Basic bounds checks
        if pivot.unique_time_idx < 0 or pivot.unique_time_idx >= n_times:
            raise ValueError(
                f"pivot.unique_time_idx out of range: {pivot.unique_time_idx}"
            )
        if pivot.global_idx < 0 or pivot.global_idx >= n_rows:
            raise ValueError(f"pivot.global_idx out of range: {pivot.global_idx}")
        if pivot.in_time_idx < 0:
            raise ValueError(f"pivot.in_time_idx must be >= 0: {pivot.in_time_idx}")

        # 2) Build the time slice for the pivot timestamp
        pivot_time = self.unique_times[pivot.unique_time_idx]
        time_slice = self.data[self.data[self.metadata.time] == pivot_time]
        if time_slice.empty:
            raise ValueError(
                f"Empty time slice for unique_time_idx={pivot.unique_time_idx}"
            )

        if pivot.in_time_idx >= len(time_slice):
            raise ValueError(
                f"pivot.in_time_idx out of range for time slice: "
                f"{pivot.in_time_idx} >= {len(time_slice)}"
            )

        # 3) Internal consistency: (unique_time_idx, in_time_idx) must map to global_idx
        expected_global_idx = int(time_slice.index[pivot.in_time_idx])
        if pivot.global_idx != expected_global_idx:
            raise ValueError(
                "Inconsistent pivot: global_idx does not match "
                f"(unique_time_idx, in_time_idx). expected={expected_global_idx}, "
                f"got={pivot.global_idx}"
            )

        # 4) Temporal consistency: row at global_idx must have the same timestamp
        row_time = self.data.iloc[pivot.global_idx][self.metadata.time]
        if row_time != pivot_time:
            raise ValueError(
                "Inconsistent pivot: row time at global_idx does not match unique_time_idx."
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ⚠️ Preencha com o caminho real do seu CSV
    # data_path = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\wildfire_pyro\\examples\\iowa_soil\\data\\ISU_Soil_Moisture_Network\\dataset_preprocessed.xlsx"
    # data_path = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\src\\wildfire_pyro\\examples\\iowa_soil\\data\\train.csv"
    data_path_windows = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\examples\\iowa_soil\\data\\daily\\processed\\dataset_with_baseline.csv"

    metadata = Metadata(
        time="valid",  # coluna de tempo
        position=["Latitude1", "Longitude1"],  # colunas espaciais
        id="station",  # coluna de identificação
        features=[
            "in_high",
            "in_low",  # temperature
            "in_rh_min",
            "in_rh",
            "in_rh_max",  # relative humidity min, avg, max
            "in_solar_mj",  # solar radiation
            "in_precip",  # preciptation
            "in_speed",  # wind speed
            # A sudden, brief increase in wind speed, typically lasting 2–5 seconds, above the mean wind speed.
            "in_gust",
            "in_et",  # evapotranspiration
            "Elevation [m]",  # elevation
        ],
        target=["out_lwmwet_1_tot"],  # , "out_lwmwet_2_tot"]  # colunas alvo
    )

    seed = 42
    rng = np.random.default_rng(seed)
    params = AdapterParams(
        min_neighborhood_size=1,
        max_neighborhood_size=4,
        max_delta_distance=1e9,
        max_delta_time=10,
        pivot_next_random=True,
        verbose=True,
    )

    adapter = DatasetAdapter(data_path_windows, metadata, params=params, seed=seed)

    # Lê uma amostra com vizinhança
    for _i in range(256):
        sample, padded, mask, feature_names, ground_truth, done = adapter.next()

    print("\n=== Padded neighbors ===")
    print(feature_names)
    print(padded)

    print("\n=== Mask ===")
    print(mask)

    print("\n=== Ground Truth ===")
    print(ground_truth)

    print("pivot:", adapter.pivot)
