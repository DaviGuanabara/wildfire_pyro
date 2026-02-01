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



class DatasetAdapter:
    "A component that maps a dataset to framework’s internal semantic model."

    def __init__(self, data_path, metadata: Metadata, params: AdapterParams, rng: np.random.Generator = np.random.default_rng(),
                 scaler: Optional[CustomScaler] = None):
        self.data_path = data_path
        self.metadata = metadata
        self.params = params

        self.load_data(self.data_path, scaler=scaler, metadata=metadata)

        self.reset(rng)

    def reset(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self._set_shapes()


        self.unique_times = np.sort(self.data[self.metadata.time].unique())
        self.cursor = self.rng.integers(int(
            self.params.max_delta_time), self.unique_times.size - int(self.params.max_delta_time))

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

            if metadata.baseline is not None:
                if len(metadata.baseline) != len(metadata.target):
                    raise ValueError(
                        "baseline and target must have the same dimensionality "
                        f"(got {len(metadata.baseline)} vs {len(metadata.target)})"
                    )
            
        elif not hasattr(self, "metadata"):
            raise ValueError("Metadata must be provided on first load.")

        self.validate(data, self.metadata)

        data["valid"] = pd.to_datetime(
            data["valid"]).map(pd.Timestamp.toordinal)


        self.data = self.sort_by_time(self.metadata, data)
        return self.data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Substitui inf e NaN por 0.0 em todas as colunas numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return data

    def _split_features_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        feature_cols = self.metadata.features
        #self.feature_cols = feature_cols
        features = df[feature_cols].to_numpy(dtype=float)
        targets = df[self.metadata.target].to_numpy(dtype=float)

        return features, targets
    
    def load_data(
        self, data_path: str, scaler: Optional[CustomScaler] = None, metadata: Optional[Metadata] = None
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
        n_neighbors = self.rng.integers(
            low=self.params.min_neighborhood_size,
            high=self.params.max_neighborhood_size + 1
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

    def _add_deltas(self, formatted: pd.DataFrame, row: pd.Series, neighbors: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
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
            raise ValueError(
                "Scaler must be initialized before adding targets.")
        
        raw = neighbors[self.metadata.target].to_numpy(dtype=float)
        scaled = self.scaler.transform_target(raw)
        new_cols = []
        for i, tgt in enumerate(self.metadata.target):
            formatted[f"{tgt}"] = scaled[:, i]
            new_cols.append(f"{tgt}")

        return formatted, new_cols

    def _add_features(self, formatted: pd.DataFrame, neighbors: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:

        if self.scaler is None:
            raise ValueError(
                "Scaler must be initialized before adding features.")
        
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
        sample = self.data.sample(n=1).iloc[0]
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

        if not hasattr(self, 'neighbor_schema'):
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
                "Scaler must be initialized before normalizing observations.")

        # 🔹 Normalize features and targets
        padded_scaled = self.scaler.transform_features(padded)
        ground_truth_scaled  = self.scaler.transform_target(ground_truth)

        return padded_scaled, ground_truth_scaled

    def _save_last_data(self, sample: pd.Series, padded_scaled, mask, feature_names, ground_truth_scaled):
        self.last_sample: pd.Series = sample
        self.last_padded = padded_scaled
        self.last_mask = mask
        self.last_feature_names = feature_names
        self.last_ground_truth = ground_truth_scaled

    def _get_sample(self, cursor) -> pd.Series:
        if cursor >= len(self.unique_times):
            raise Exception("[dataset_adapter] no more dates.")

        current_time = self.unique_times[cursor]
        time_slice = self.data[self.data[self.metadata.time] == current_time]

        # Escolhe um índice aleatório do subconjunto de linhas daquele timestamp
        chosen_idx = self.rng.choice(time_slice.index)
        sample = time_slice.loc[chosen_idx]
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


    def _read(self, cursor) -> Tuple[pd.Series, np.ndarray, np.ndarray, List[str], np.ndarray, bool]:
        """
        Return next row with its neighborhood (padded).
        Iterates sequentially and flags `done=True` when the dataset ends.
        Normalizes features and targets using the configured scaler.
        """

        if self.scaler is None:
            raise ValueError(
                "Scaler must be initialized before transform targets.")
        
        
        # Verifica antes de ler
        if self.cursor >= len(self.unique_times) - 1:
            self.done = True

        sample = self._get_sample(cursor)
        row_index = sample.name
        
        neighbors = self.get_neighbors(row_index=cast(int, row_index), row=sample)

        formatted = self.format_neighbors(sample, neighbors)

        feature_names = list(formatted.columns)
        ground_truth = self.get_ground_truth(sample)
        ground_truth_scaled = self.scaler.transform_target(ground_truth.reshape(1, -1)).flatten()

        padded, mask = self.pad_neighbors(
            formatted, max_neighborhood_size=self.params.max_neighborhood_size
        )

        #padded_scaled, ground_truth_scaled = self.normalize_observation(padded, ground_truth)
        self._save_last_data(sample, padded, mask, feature_names, ground_truth_scaled)
        
        return sample, padded, mask, feature_names, ground_truth_scaled, self.done


    def read_resample_neighbors(self):
        return self._read(self.cursor)

    def next(self) -> Tuple[pd.Series, np.ndarray, np.ndarray, List[str], np.ndarray, bool]:
        """
        Return next row with its neighborhood (padded).
        Iterates sequentially and flags `done=True` when the dataset ends.
        """

        reading = self._read(self.cursor)
        self.cursor += 1  # Só avança depois de ler com sucesso
        return reading


    def last(self) -> Tuple[pd.Series, np.ndarray, np.ndarray, List[str], np.ndarray, bool]:
        """Returns the last sample returned by `next()`."""
        return (self.last_sample, self.last_padded, self.last_mask,
                self.last_feature_names, self.last_ground_truth, self.done)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ⚠️ Preencha com o caminho real do seu CSV
    # data_path = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\wildfire_pyro\\examples\\iowa_soil\\data\\ISU_Soil_Moisture_Network\\dataset_preprocessed.xlsx"
    #data_path = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\src\\wildfire_pyro\\examples\\iowa_soil\\data\\train.csv"
    data_path_windows = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\examples\\iowa_soil\\data\\daily\\processed\\dataset_with_baseline.csv"



    metadata = Metadata(
        time="valid",  # coluna de tempo
        position=["Latitude1", "Longitude1"],  # colunas espaciais
        id="station",  # coluna de identificação
        features=[
            "in_high", "in_low", #temperature
            "in_rh_min", "in_rh", "in_rh_max", #relative humidity min, avg, max
            "in_solar_mj", #solar radiation
            
            "in_precip", #preciptation
            "in_speed", #wind speed
            # A sudden, brief increase in wind speed, typically lasting 2–5 seconds, above the mean wind speed.
            "in_gust",
            "in_et", #evapotranspiration
            "Elevation [m]", #elevation
        ],
        target=["out_lwmwet_1_tot"],  # , "out_lwmwet_2_tot"]  # colunas alvo
    )

    params = AdapterParams(
        min_neighborhood_size=1,
        max_neighborhood_size=4,
        max_delta_distance=1e9,
        max_delta_time=10,
        verbose=True,
    )

    adapter = DatasetAdapter(data_path_windows, metadata, params=params)

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

    print("cursor:", adapter.cursor)