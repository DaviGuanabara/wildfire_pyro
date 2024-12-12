import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


class SensorManager:

    TIME_TAG = "t"
    LATITUDE_TAG = "lat"
    LONGITUDE_TAG = "lon"
    SENSOR_ID_TAG = "sensor_id"

    def __init__(self, data_path):
        """
        Initialize the SensorManager with the dataset.

        Args:
            data_path (str): Path to the dataset file.
        """
        self.data = pd.read_csv(data_path)
        self.data = self.data.sort_values(by="t").reset_index(
            drop=True
        )  
        self.data["sensor_id"] = self.data.groupby(["lat", "lon"]).ngroup()
        self.sensors = self.data["sensor_id"].unique()
        self.current_sensor = None
        self.data_from_current_sensor = None
        self.current_time_index = 0 

    def set_random_time(self):
        """
        Set the current time index to a random valid position within the sensor's data.

        Returns:
            int: The selected random time index.
        """
        if self.current_sensor is None or self.data_from_current_sensor is None:
            raise ValueError("No sensor selected. Call `select_sensor()` first.")

        # Randomly select an index within the current sensor's data range
        self.current_time_index = np.random.randint(len(self.data_from_current_sensor))
        return self.current_time_index

    def set_random_sensor(self):
        """
        Randomly select a sensor and update its corresponding data.
        """
        self.current_sensor = np.random.choice(self.sensors)
        self.data_from_current_sensor = self.data[
            self.data["sensor_id"] == self.current_sensor
        ]

        self.set_random_time()

    def increment_time(self):
        """
        Increment the current time to the next available value for the current sensor.
        """
        if self.current_sensor is None:
            raise ValueError("No sensor selected. Call `update_sensor()` first.")

        if self.current_time_index + 1 < len(self.data_from_current_sensor):
            self.current_time_index += 1
        else:
            logging.warning("No more readings available for the current sensor. Sensor will be changed")

            # tempo atual
            current_time = self.data_from_current_sensor.iloc[self.current_time_index]["t"]
            # filtrar sensores a partir desse tempo
            future_data = self.data[self.data["t"] > current_time]
            # ordenar dados pelo tempo
            future_data = future_data.sort_values(by="t").reset_index(drop=True)
            # escolher o tempo de menor valor dentro do novo conjunto de dados

            sensor_id = future_data.iloc[0]["sensor_id"]
            current_time = future_data.iloc[0]["t"]

            self.current_sensor = sensor_id
            self.data_from_current_sensor = (
                self.data[self.data["sensor_id"] == self.current_sensor]
                .sort_values(by="t")
                .reset_index(drop=True)
            )
            self.current_time_index = 0

    def get_reading(self) -> pd.Series:
        """
        Obtém a leitura para o sensor atual no índice de tempo atual.

        Returns:
            pd.Series: A linha de dados correspondente ao índice de tempo atual, sem 'sensor_id'.
        """
        if self.current_sensor is None:
            raise ValueError("No sensor selected. Call `update_sensor()` first.")

        reading = self.data_from_current_sensor.iloc[self.current_time_index].drop('sensor_id')
        return reading


    def get_neighbors(self, n_neighbors_max: int, n_neighbors_min: int = 1, time_window=-1, distance_window = -1):
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

        step = self.current_time_index
        current_time = self.data_from_current_sensor.iloc[step][self.TIME_TAG]
        sensor_id = self.current_sensor

        start_time = 0 if time_window == -1 else current_time - time_window

        # Filtrar os dados dentro da janela de tempo
        windowned_data = self.data[
            (self.data[self.TIME_TAG] >= start_time)
            & (self.data[self.TIME_TAG] <= current_time)
        ]

        # Excluir o sensor avaliado da lista de possíveis vizinhos
        possible_neighbors_data = windowned_data[
            windowned_data[self.SENSOR_ID_TAG] != sensor_id
        ]

        possible_neighbors = possible_neighbors_data[self.SENSOR_ID_TAG].unique()

        if len(possible_neighbors) == 0:
            logging.warning("Nenhum sensor disponível para ser vizinho.")
            return pd.DataFrame([])

        num_neighbors = np.random.randint(n_neighbors_min, n_neighbors_max + 1)

        # Selecionar aleatoriamente os sensores para serem os vizinhos
        # Se o número de vizinhos for maior que o número de sensores disponíveis, escolher com reposição
        neighbor_sensors = np.random.choice(
            possible_neighbors,
            size=num_neighbors,
            replace=num_neighbors > len(possible_neighbors),
        )  # True)

        selected_neighbors = []

        """
        for neighbor_id in neighbor_sensors:

            neighbor_data = windowned_data[windowned_data[self.SENSOR_ID_TAG] == neighbor_id]

            if neighbor_data.empty:
                continue  

            random_index = np.random.choice(neighbor_data.index)
            selected_neighbors.append(neighbor_data.loc[random_index])
        """
        selected_neighbors = pd.concat(
            [
                possible_neighbors_data[
                    possible_neighbors_data[self.SENSOR_ID_TAG] == neighbor_id
                ].sample(1)
                for neighbor_id in neighbor_sensors
            ]
        ).reset_index(drop=True)

        neighbors_df = selected_neighbors.drop(columns=['sensor_id'])

        return neighbors_df

    def get_position(self):
        """
        Get the latitude and longitude of the current sensor.

        Returns:
            tuple: (latitude, longitude) of the current sensor.
        """
        if self.current_sensor is None:
            raise ValueError("No sensor selected. Call `update_sensor()` first.")

        # A latitude e longitude são constantes para o sensor atual
        lat = self.data_from_current_sensor[self.LATITUDE_TAG].iloc[0]
        lon = self.data_from_current_sensor[self.LONGITUDE_TAG].iloc[0]
        return lat, lon

    #IN DEVELOPMENT
    def find_sensors_in_region(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> list:
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
            (self.data[self.LATITUDE_TAG] >= lat_min) &
            (self.data[self.LONGITUDE_TAG] >= lon_min) &
            (self.data[self.LATITUDE_TAG] <= lat_max) &
            (self.data[self.LONGITUDE_TAG] <= lon_max)
        ]
        sensors_in_region = region_data[self.SENSOR_ID_TAG].unique().tolist()
        logging.info(f"{len(sensors_in_region)} sensores encontrados na região especificada.")
        return sensors_in_region
