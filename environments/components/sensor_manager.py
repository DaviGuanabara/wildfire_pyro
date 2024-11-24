import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

#NÃO ESTÁ FUNCIONANDO! O SCRIPT VER.PY ESTÁ FUNCIONANDO CORRETAMENTE. OLHAR ELE
logging.warning("TENHO QUE CORRIGIR ESSE SENSORMANAGER. OLHAR O SCRIPT VER.PY")
    
class SensorManager:
    def __init__(self, data_path):
        """
        Initialize the SensorManager with the dataset.

        Args:
            data_path (str): Path to the dataset file.
        """
        self.data = pd.read_csv(data_path)
        self.data = self.data.sort_values(by="t").reset_index(
            drop=True
        )  # Ordenar por tempo
        self.data["sensor_id"] = self.data.groupby(["lat", "lon"]).ngroup()
        self.sensors = self.data["sensor_id"].unique()
        self.current_sensor = None
        self.current_sensor_data = None
        self.current_time_index = None  # Índice do tempo atual

    
    #O QUE TÁ CAGANDO AQUI É ESSE CURRENT_SENSOR_DATA. AÍ FICA PRESO SOMENTE NA LEITURA DO SENSOR ATUAL, QUE TALVEZ NEM TENHA VIZINHO.
    def set_random_time(self):
        """
        Set the current time index to a random valid position within the sensor's data.

        Returns:
            int: The selected random time index.
        """
        if self.current_sensor is None or self.current_sensor_data is None:
            raise ValueError("No sensor selected. Call `select_sensor()` first.")

        # Randomly select an index within the current sensor's data range
        self.current_time_index = np.random.randint(len(self.current_sensor_data))
        return self.current_time_index

    def update_sensor(self):
        """
        Randomly select a sensor and update its corresponding data.
        """
        self.current_sensor = np.random.choice(self.sensors)
        self.current_sensor_data = self.data[
            self.data["sensor_id"] == self.current_sensor
        ]
        self.current_time_index = (
            0  # Reiniciar para o primeiro tempo do sensor selecionado
        )

    def increment_time(self):
        """
        Increment the current time to the next available value for the current sensor.
        """
        if self.current_sensor is None:
            raise ValueError("No sensor selected. Call `update_sensor()` first.")

        if self.current_time_index + 1 < len(self.current_sensor_data):
            self.current_time_index += 1
        else:
            raise IndexError("No more readings available for the current sensor.")

    def get_reading(self):
        """
        Get the reading for the current sensor at the global current time index.

        Returns:
            pd.Series: The data row corresponding to the current time index.
        """
        if self.current_sensor is None:
            raise ValueError("No sensor selected. Call `update_sensor()` first.")

        return self.current_sensor_data.iloc[self.current_time_index]

    #NÃO ESTÁ FUNCIONANDO! O SCRIPT VER.PY ESTÁ FUNCIONANDO CORRETAMENTE. OLHAR ELE
    logging.warning("A função get_neighbors não está funcionando corretamente.")
    def get_neighbors(self, step, time_window, n_neighbors_min: int, n_neighbors_max: int):
        """
        Get neighboring readings within a time window.

        Args:
            step (int): The current step in the environment.
            time_window (float): Time window for selecting neighbors.
            n_neighbors_min (int): Minimum number of neighbors to select.
            n_neighbors_max (int): Maximum number of neighbors to select.

        Returns:
            pd.DataFrame: A DataFrame with the neighboring readings.
        """
        if self.current_sensor is None:
            raise ValueError("No sensor selected. Call `select_sensor()` first.")

        # Get the current time and exclude the current row (sensor being evaluated)
        current_time = self.current_sensor_data.iloc[step]["t"]
        time_range_start = max(current_time - time_window, self.current_sensor_data["t"].min())

        # Filter neighbors based on the time window and exclude the current sensor reading
        neighbors = self.current_sensor_data[
            (self.current_sensor_data["t"] >= time_range_start)
            & (self.current_sensor_data["t"] < current_time)
            & (self.current_sensor_data.index != self.current_sensor_data.index[step])
        ]

        # Handle case where not enough neighbors are available
        if len(neighbors) < n_neighbors_min:
            logging.warning(
                f"Fewer neighbors ({len(neighbors)}) than minimum requested ({n_neighbors_min}). Returning all available neighbors."
            )
            return neighbors

        # Randomly sample a number of neighbors between n_neighbors_min and n_neighbors_max
        n_neighbors = np.random.randint(n_neighbors_min, n_neighbors_max + 1)
        if n_neighbors > len(neighbors):
            n_neighbors = len(neighbors)

        return neighbors.sample(n_neighbors)


    def get_position(self):
        """
        Get the latitude and longitude of the current sensor.

        Returns:
            tuple: (latitude, longitude) of the current sensor.
        """
        if self.current_sensor is None:
            raise ValueError("No sensor selected. Call `update_sensor()` first.")

        # A latitude e longitude são constantes para o sensor atual
        lat = self.current_sensor_data["lat"].iloc[0]
        lon = self.current_sensor_data["lon"].iloc[0]
        return lat, lon

    
    def get_all_neighbors(self):
        """
        Get all neighboring readings for the current sensor, excluding the current reading.

        Returns:
            pd.DataFrame: A DataFrame with all neighboring readings for the current sensor.
        """
        if self.current_sensor is None or self.current_sensor_data is None:
            raise ValueError("No sensor selected. Call `update_sensor()` first.")
        
        if self.current_time_index is None:
            raise ValueError("Current time index is not set. Call `set_random_time()` or use `increment_time()` first.")

        # Exclude the current reading from the neighbors
        neighbors = self.current_sensor_data[
            self.current_sensor_data.index != self.current_sensor_data.index[self.current_time_index]
        ]

        if neighbors.empty:
            logging.warning("No neighbors available for the current sensor.")
            return pd.DataFrame(columns=self.current_sensor_data.columns)

        return neighbors
