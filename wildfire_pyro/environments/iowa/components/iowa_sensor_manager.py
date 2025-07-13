from typing import Tuple, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("SensorManager")
logger.setLevel(logging.INFO)


class IOWASensorManager:

    TIME_TAG = "t"
    LATITUDE_TAG = "lat"
    LONGITUDE_TAG = "lon"
    SENSOR_ID_TAG = "sensor_id"
    SENSOR_FEATURES = ["t", "lat", "lon", "y"]


    def __init__(self, data_path, verbose: bool = False):
        """
        Initialize the SensorManager with the dataset.

        Args:
            data_path (str): Path to the dataset file.
        """
        self.data = pd.read_csv(data_path)


        # Após ler o CSV
        #reorder columns to have 't' first if it exists
        cols = list(self.data.columns)
        if self.TIME_TAG in cols:
            cols.insert(0, cols.pop(cols.index(self.TIME_TAG)))
            self.data = self.data[cols]


        self.data = self.data.sort_values(by="t").reset_index(drop=True)

        self.data["sensor_id"] = self.data.groupby(["lat", "lon"]).ngroup()
        self.sensors = self.data["sensor_id"].unique()


        self.verbose = verbose

        # Build a lookup map for fast access to each sensor's data
        self.sensor_data_map: dict[int, pd.DataFrame] = {
            int(sensor_id): df.reset_index(drop=True) # type: ignore
            for sensor_id, df in self.data.groupby("sensor_id", observed=True)

        }

        # Liga sensor IDs a posições no DataFrame
        self.sensor_index = {
            sid: df_sdf.index.tolist()
            for sid, df_sdf in self.data.groupby("sensor_id", sort=False)
        }

        self.reset()

    def init_tags(self):


        self.tags = {

            "time": "data",
            "latitude": "Latitude1",
            "longitude": "Longitude1",
            "altitude": "Elevation [m]",  
            "id": "ID",
            "target": "y",
            "features": [ "high", "low", "rh_min", "rh", "rh_max", "solar_mj", "precip", "speed", "gust", "et",
        "lwmv_1", "lwmv_2", "lwmdry_1_tot", "lwmcon_1_tot", "lwmwet_1_tot",
        "lwmdry_2_tot", "lwmcon_2_tot", "lwmwet_2_tot", "bpres_avg"]
        }



    def reset(self, seed: int = 0):
        """
        Reset the SensorManager to its initial state.

        Args:
            seed (int): The seed to use for the random number generator.
        """

        self.rng = np.random.default_rng(seed)

        self.state_tracker = {
            "current_step": 0,  # Passo atual da simulação
            "neighbors_step": -1,  # Último passo em que vizinhos foram calculados
            "deltas_step": -1,  # Último passo em que deltas foram calculados
            "current_time_index": 0,  # Índice de tempo atual
            "current_sensor": None,  # Sensor atual
            "bootstrap_neighbors_step": -1,
        }

        self.cache: dict[str, Any] = {
            "neighbors": None,  # Dados calculados dos vizinhos
            "deltas": None,  # Dados calculados dos deltas
            "data_from_current_sensor": None,  # Dados do sensor atual
            "current_reading": None,  # Novo: armazena a leitura atual do sensor
            "ground_truth": None,  # Novo: armazena o ground truth diretamente
            "bootstrap_neighbors": None,
        }

        
        self._select_random_sensor()

    def set_random_time_index(self):
        """
        Set the current time index to a random valid position within the sensor's data.

        """

        # Randomly select an index within the current sensor's data range
        self.state_tracker["current_time_index"] = self.rng.integers(
            len(self.cache["data_from_current_sensor"])
        )

    def _select_random_sensor(self):
        """
        Selects a random sensor from the dataset.

        Returns:
            int: The randomly selected sensor ID.
        """
        self.state_tracker["current_sensor"] = self.rng.choice(self.sensors)
        self.cache["data_from_current_sensor"] = self.sensor_data_map[
            self.state_tracker["current_sensor"]
        ]

        self.set_random_time_index()

        

    def step(self):
        """
        Randomly select a sensor and update its corresponding data.
        """

        self._select_random_sensor()

        # Incrementa o passo atual para controle de cache
        self.state_tracker["current_step"] += 1

    def get_current_sensor_data(self) -> pd.Series:
        """
        Obtém a leitura para o sensor atual no índice de tempo atual.

        Returns:
            pd.Series: A linha de dados correspondente ao índice de tempo atual, sem 'sensor_id'.
        """
        if self.state_tracker["current_sensor"] is None:
            raise ValueError("No sensor selected. Call `reset()` first.")


        sensor_df = self.cache.get("data_from_current_sensor")
        if sensor_df is None:
            raise RuntimeError("Sensor data is not initialized in cache.")

        #reading = sensor_df.iloc[self.state_tracker["current_time_index"]].drop(
        #    "sensor_id")
        
        #TODO: CORRIGIR O NOME E O USO DO SENSOR FEATURES
        #SENSOR FEATURES ESTÁ SENDO ERRONEAMENTE NOMEADO. É COMO SE FOSSE TODAS AS COLUNAS, E NÃO SÓ
        # as colunas de interesse do sensor.
        reading = sensor_df.loc[self.state_tracker["current_time_index"],
                                self.SENSOR_FEATURES]



        # Salva a leitura atual e o ground truth no cache
        self.cache["current_reading"] = reading
        self.cache["ground_truth"] = reading["y"]

        return reading

    def get_ground_truth(self) -> float:
        """
        Retorna o ground truth (valor real de 'y') do sensor atual.
        """
        if self.cache["ground_truth"] is None:
            self.get_current_sensor_data()  # Garante que a leitura atual está carregada

        return self.cache["ground_truth"]

    def get_neighbors(
        self,
        n_neighbors_max: int,
        n_neighbors_min: int = 1,
        time_window=-1,
        distance_window=-1,
    ):
        """
        Seleciona vizinhos aleatórios para um sensor específico dentro de uma janela de tempo.

        :param n_neighbors_min: Número mínimo de vizinhos (padrão=1)
        :param n_neighbors_max: Número máximo de vizinhos
        :param time_window: Janela de tempo (número de passos antes)
        :param distance_window: Janela de distância (não utilizado atualmente)
        :return: DataFrame com os vizinhos selecionados
        """
        if n_neighbors_min > n_neighbors_max:
            raise ValueError("n_neighbors_min não pode ser maior que n_neighbors_max.")

        if self.state_tracker["neighbors_step"] == self.state_tracker["current_step"]:
            return self.cache["neighbors"]

        neighbors = self._compute_neighbors(
            n_neighbors_min, n_neighbors_max, time_window=time_window, distance_window=distance_window
        )

        self.cache["neighbors"] = neighbors
        self.state_tracker["neighbors_step"] = self.state_tracker["current_step"]

        return neighbors

    def _compute_neighbors(
        self, n_neighbors_min, n_neighbors_max, time_window, distance_window
    ) -> pd.DataFrame:
        """
        Computes the neighbors for the current sensor within a specified time window.
        """
        # Obter informações básicas do sensor atual
        sensor_id = self.state_tracker["current_sensor"]
        timestamp = self._get_current_timestamp()

        # Filtrar dados na janela de tempo e remover o sensor atual
        candidate_neighbors = self._filter_candidates(
            timestamp=timestamp, time_window=time_window, sensor_id=sensor_id, distance_window=distance_window)

        if candidate_neighbors.empty:
            logger.info(f"No neighbors found for the sensor {sensor_id}.")
            return pd.DataFrame([])

        # Selecionar aleatoriamente os vizinhos
        num_neighbors = self.rng.integers(n_neighbors_min, n_neighbors_max + 1)
        selected_neighbors = self._select_random_neighbors(
            candidate_neighbors, num_neighbors
        )

        return selected_neighbors.drop(columns=[self.SENSOR_ID_TAG])

    # Subfunções auxiliares mais coesas
    def _get_current_timestamp(self) -> float:
        """Returns the current timestamp based on the sensor's current time index."""
        current_index = self.state_tracker["current_time_index"]
        return self.cache["data_from_current_sensor"].iloc[current_index][self.TIME_TAG]

    def _filter_candidates(
        self, sensor_id: int, timestamp: float, time_window: int = -1, distance_window: float = -1
    ) -> pd.DataFrame:
        """
        Filters potential neighbors within the specified time window,
        excluding the current sensor.
        """
        start_time = 0 if time_window == -1 else timestamp - time_window


        mask = self.data[self.TIME_TAG].between(start_time, timestamp)
        windowed_data = self.data.loc[mask]

        # Aplica filtro espacial se necessário
        if distance_window > 0:
            ref_lat, ref_lon = self.get_current_sensor_position()
            d_lat = windowed_data[self.LATITUDE_TAG] - ref_lat
            d_lon = windowed_data[self.LONGITUDE_TAG] - ref_lon
            distance = np.sqrt(d_lat**2 + d_lon**2)
            windowed_data = windowed_data[distance <= distance_window]



        return windowed_data[windowed_data[self.SENSOR_ID_TAG] != sensor_id]

    def _select_random_neighbors(self, candidates: pd.DataFrame, num_neighbors: int) -> pd.DataFrame:
        # Pega sensor IDs disponíveis
        sids = candidates['sensor_id'].unique()
        selected_sids = self.rng.choice(
            sids, size=num_neighbors, replace=(num_neighbors > len(sids)))

        rows = []
        for sid in selected_sids:
            idxs = self.sensor_index.get(sid)
            if not idxs:
                continue
            i = self.rng.integers(len(idxs))
            rows.append(self.data.iloc[idxs[i]])

        return pd.DataFrame(rows).reset_index(drop=True)


    def ensure_minimum_neighbors(
        self, n_neighbors_max, n_neighbors_min, time_window, distance_window
    ) -> pd.DataFrame:
        """
        Ensures the minimum number of neighbors is met. If not enough neighbors are found,
        steps through the environment until the requirement is met.

        Args:
            n_neighbors_max (int): Maximum number of neighbors.
            n_neighbors_min (int): Minimum number of neighbors.
            time_window (int): Time window for neighbor search.
            distance_window (int): Distance window (currently unused).

        Returns:
            pd.DataFrame: DataFrame with the found neighbors. May be empty if no valid neighbors are found.
        """
        if not hasattr(self, "sensors_without_neighbors"):
            self.sensors_without_neighbors = set()

        while True:
            current_sensor_id = int(
                self.state_tracker["current_sensor"]
            )  # Ensure it's a regular int

            # Skip sensors already identified as having no neighbors
            if current_sensor_id in self.sensors_without_neighbors:
                self.step()
                continue

            neighbors = self.get_neighbors(
                n_neighbors_max, n_neighbors_min, time_window, distance_window
            )

            if len(neighbors) >= n_neighbors_min:
                return neighbors  # Sufficient neighbors found

            # Add sensor to the "no neighbors" list and log the update
            self.sensors_without_neighbors.add(current_sensor_id)

            if self.verbose:
                logger.info(
                    f"Sensors without enough neighbors: {sorted(self.sensors_without_neighbors)}"
                )

            self.step()  # Move to the next sensor

    def get_neighbors_deltas(
        self,
        n_neighbors_max: int,
        n_neighbors_min: int = 1,
        time_window=-1,
        distance_window=-1,
    ) -> pd.DataFrame:
        """
        Calculates the deltas between the current sensor and its neighbors.

        Returns:
            pd.DataFrame: Calculated deltas for each variable.
        """
        if self.state_tracker["deltas_step"] == self.state_tracker["current_step"]:
            return self.cache["deltas"]

        reference_sensor: pd.Series = self.get_current_sensor_data()
        neighbors_sensors: pd.DataFrame = self.ensure_minimum_neighbors(
            n_neighbors_max, n_neighbors_min, time_window, distance_window
        )

        deltas = self._compute_deltas(neighbors_sensors, reference_sensor)

        self.cache["deltas"] = deltas
        self.state_tracker["deltas_step"] = self.state_tracker["current_step"]

        return deltas

    def _compute_deltas(
        self, neighbors: pd.DataFrame, reference: pd.Series
    ) -> pd.DataFrame:
        """
        Computes the deltas between the reference sensor and its neighbors,
        preserving the 'y' value from each neighbor.

        Args:
            neighbors (pd.DataFrame): Data from neighboring sensors.
            reference (pd.Series): Data from the reference (current) sensor.

        Returns:
            pd.DataFrame: Calculated deltas for each variable, with 'y' preserved from the neighbors.
        """

        # Variables to compute deltas (excluding 'y')
        delta_columns = [col for col in neighbors.columns if col != "y"]

        # Delta calculation: neighbors - reference
        ref_array = reference[delta_columns].to_numpy()


        delta_array = neighbors[delta_columns].to_numpy() - ref_array
        deltas = pd.DataFrame(delta_array, columns=delta_columns,
                            index=neighbors.index)
        
        # preserve the 'y' value from neighbors
        deltas["y"] = neighbors["y"]


        return deltas

    def get_current_sensor_position(self):
        """
        Get the latitude and longitude of the current sensor.

        Returns:
            tuple: (latitude, longitude) of the current sensor.
        """
        if self.state_tracker["current_sensor"] is None:
            raise ValueError("No sensor selected. Call `step()` first.")

        # lat and lon are constant
        lat = self.cache["data_from_current_sensor"][self.LATITUDE_TAG].iloc[0]
        lon = self.cache["data_from_current_sensor"][self.LONGITUDE_TAG].iloc[0]
        return lat, lon

    def get_current_sensor_time(self):
        """
        Obtém o tempo atual do sensor com base no índice de tempo atual.

        Returns:
            float: O tempo atual correspondente ao índice de tempo do sensor selecionado.
        """
        if self.state_tracker["current_sensor"] is None:
            raise ValueError("Nenhum sensor selecionado. Chame `step()` primeiro.")

        # Obtain the value of the time based on the index of current time
        i = self.state_tracker["current_time_index"]
        sensor_df = self.cache["data_from_current_sensor"]
        current_time = sensor_df.iloc[i][self.TIME_TAG]

        return current_time

    # IN DEVELOPMENT
    def find_sensors_in_region(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> list:
        """
        Encontra sensores dentro de uma região geográfica especificada.

        Args:
            lat_min (float): Latitude mínima.
            lat_max (float): Latitude máxima.
            lon_min (float): Longitude mínima.
            lon_max (float): Longitude máxima.

        Returns:
            list: Lista de sensor_ids que estão dentro da região especificada.
        """
        region_data = self.data[
            (self.data[self.LATITUDE_TAG] >= lat_min)
            & (self.data[self.LONGITUDE_TAG] >= lon_min)
            & (self.data[self.LATITUDE_TAG] <= lat_max)
            & (self.data[self.LONGITUDE_TAG] <= lon_max)
        ]
        sensors_in_region = region_data[self.SENSOR_ID_TAG].unique().tolist()
        logger.info(
            f"{len(sensors_in_region)} sensores encontrados na região especificada."
        )
        return sensors_in_region

    def get_bootstrap_neighbors(
        self,
        n_bootstrap: int = 20,
        n_neighbors_max: int = 5,
        n_neighbors_min: int = 2,
        time_window: int = -1,
        distance_window: int = -1,
        force_recompute: bool = True,
    ) -> list:
        """
        For a locked sensor, generate multiple sets of neighbors (e.g., 20 sets) using the
        underlying random selection. This method bypasses the standard caching used in
        get_neighbors so that each call can return a different set.

        Args:
            n_bootstrap (int): Number of neighbor sets to generate.
            n_neighbors_max (int): Maximum number of neighbors.
            n_neighbors_min (int): Minimum number of neighbors.
            time_window (int): Time window for neighbor search.
            distance_window (int): Distance window (unused currently).
            force_recompute (bool): If True, regenerate even if cached.

        Returns:
            list: A list of pd.DataFrame objects, each containing a set of neighbors.
        """
        current_step = self.state_tracker["current_step"]

        # Check if bootstrap neighbors exist and correspond to the current step.
        if (
            not force_recompute
            and self.cache.get("bootstrap_neighbors") is not None
            and self.cache.get("bootstrap_neighbors_step") == current_step
        ):
            return self.cache["bootstrap_neighbors"]

        bootstrap_neighbors = []
        # Loop to generate n_bootstrap different neighbor sets.
        for _ in range(n_bootstrap):
            # Call _compute_neighbors directly (bypassing the cache in get_neighbors)
            neighbors = self._compute_neighbors(
                n_neighbors_min=n_neighbors_min, n_neighbors_max=n_neighbors_max, time_window=time_window, distance_window=distance_window
            )
            bootstrap_neighbors.append(neighbors)

        # Store the generated neighbor sets and the current step in the cache.
        self.cache["bootstrap_neighbors"] = bootstrap_neighbors
        self.cache["bootstrap_neighbors_step"] = current_step

        return bootstrap_neighbors

    def get_bootstrap_neighbors_deltas(
        self,
        n_bootstrap: int = 20,
        n_neighbors_max: int = 5,
        n_neighbors_min: int = 2,
        time_window: int = -1,
        distance_window: int = -1,
        force_recompute: bool = True,
    ) -> Tuple[list, float]:
        """
        For a locked sensor, generate multiple sets of neighbor deltas using bootstrap.

        This method calculates the deltas for each bootstrap neighbor set (using the target sensor as reference)
        and returns a list of delta DataFrames along with a single ground truth value (the 'y' of the target sensor).

        Args:
            n_bootstrap (int): Number of bootstrap samples to generate.
            n_neighbors_max (int): Maximum number of neighbors.
            n_neighbors_min (int): Minimum number of neighbors.
            time_window (int): Time window for neighbor search.
            distance_window (int): Distance window (unused currently).
            force_recompute (bool): If True, force regeneration even if cached.

        Returns:
            Tuple[list, float]:
                - A list of pd.DataFrame objects, each containing the computed deltas for one bootstrap sample.
                - A single ground truth value corresponding to the target sensor's 'y' value.
        """
        # Retrieve the target sensor data (the sensor remains locked until the next step)
        target_sensor = self.get_current_sensor_data()
        ground_truth = target_sensor["y"]

        # Get a list of bootstrap neighbor sets
        bootstrap_neighbors_list = self.get_bootstrap_neighbors(
            n_bootstrap=n_bootstrap,
            n_neighbors_max=n_neighbors_max,
            n_neighbors_min=n_neighbors_min,
            time_window=time_window,
            distance_window=distance_window,
            force_recompute=force_recompute,
        )

        bootstrap_deltas = []
        # For each bootstrap neighbor set, compute the deltas with the target sensor as reference
        for neighbors in bootstrap_neighbors_list:
            deltas = self._compute_deltas(neighbors, target_sensor)
            bootstrap_deltas.append(deltas)

        return bootstrap_deltas, ground_truth
