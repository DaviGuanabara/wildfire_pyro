import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional
from gymnasium import spaces, Env

from wildfire_pyro.environments.components.sensor_manager import SensorManager
from wildfire_pyro.environments.base_environment import BaseEnvironment

# Configuração básica do logging
logging.basicConfig(level=logging.INFO)


class FixedSensorEnvironment(BaseEnvironment):
    def __init__(
        self,
        data_path: str,
        max_steps: int = 50,
        n_neighbors_min: int = 5,
        n_neighbors_max: int = 10,
    ):
        """
        Inicializa o Fixed Sensor Environment.

        Args:
            data_path (str): Caminho para o dataset.
            max_steps (int): Número máximo de passos por episódio.
            n_neighbors_min (int): Número mínimo de vizinhos.
            n_neighbors_max (int): Número máximo de vizinhos.
        """
        super(FixedSensorEnvironment, self).__init__()

        self.max_steps = max_steps
        self.n_neighbors_min = n_neighbors_min
        self.n_neighbors_max = n_neighbors_max
        self.current_step = 0
        self.ground_truth = None

        # Inicializar SensorManager (já realiza o pré-processamento dos dados)
        self.sensor_manager: SensorManager = SensorManager(data_path)

        # Definir espaços de observação e ação
        self._define_spaces()

    def _define_spaces(self):
        """
        Define os espaços de observação e ação.
        """
        # Observação: (n_neighbors_max, 5) - 4 para u_matrix + 1 para mask
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_neighbors_max, 5), dtype=np.float32
        )

        # Ação: Um único valor contínuo
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reseta o ambiente para um novo episódio.

        Args:
            seed (Optional[int]): Semente para randomização.
            options (Optional[dict]): Opções adicionais.

        Returns:
            Tuple[np.ndarray, dict]: Observação inicial e informações adicionais.
        """
        self.current_step = 0
        
        self.sensor_manager.reset(seed)

        observation, self.ground_truth = self._generate_observation()

        return observation, {"ground_truth": self.ground_truth, "sensor": self._sensor_info()}

    #TODO: a ação deve mesmo ser um array ?
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        A step in the environment.

        Args:
            action (np.ndarray): Action taken by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: Observation, reward, terminated, truncated, info.
        """
        
        
        
        self.current_step += 1
        self.sensor_manager.step()

        observation, self.ground_truth = self._generate_observation()
        reward = self._calculate_reward()

        terminated = self._is_terminated()
        truncated = False 
        
        return (
            observation,
            reward,
            terminated,
            truncated,
            {"ground_truth": self.ground_truth, "sensor": self._sensor_info()},
        )

    def _sensor_info(self) -> dict:
        """
        Retrieves information about the current sensor.

        Returns:
            dict: Contains latitude, longitude, timestamp, and sensor ID.
        """
        sensor_id = self.sensor_manager.state_tracker["current_sensor"]
        current_index = self.sensor_manager.state_tracker["current_time_index"]
        sensor_data = self.sensor_manager.cache["data_from_current_sensor"].iloc[current_index]

        sensor_info = {
            "lat": round(sensor_data[self.sensor_manager.LATITUDE_TAG], 4),
            "lon": round(sensor_data[self.sensor_manager.LONGITUDE_TAG], 4),
            "t": round(sensor_data[self.sensor_manager.TIME_TAG], 4),
            "sensor_id": sensor_id,
        }

        return sensor_info


    def _calculate_reward(self) -> float:

        return None

    def _is_terminated(self) -> bool:
        """
        Verifica se o episódio deve terminar.

        Returns:
            bool: True se o episódio deve terminar, False caso contrário.
        """
        if self.current_step >= self.max_steps:
            # logging.info(
            #    "Número máximo de passos alcançado. Episódio terminando.")
            return True
        return False

    def _generate_observation(self) -> Tuple[np.ndarray, float]:
        """
        Gera a observação e o ground truth para o passo atual.

        Returns:
            Tuple[np.ndarray, float]: Observação com shape (n_neighbors_max, 5) e ground truth.
        """
        # Obtém os deltas dos vizinhos já processados
        deltas = self.sensor_manager.get_neighbors_deltas(self.n_neighbors_max, self.n_neighbors_min)

        # Criação da matriz de observação (n_neighbors_max, 4)
        #TODO: Esse hardcoded 4 é o tamanho do vetor de deltas. Deveria ser um parâmetro
        #retirado da própria classe SensorManager, da base de dados.
        observation_matrix = np.zeros((self.n_neighbors_max, 4), dtype=np.float32)
        mask = np.zeros(self.n_neighbors_max, dtype=bool)

        # Preenche a matriz com os deltas existentes
        num_neighbors = len(deltas)
        observation_matrix[:num_neighbors, :] = deltas.values
        mask[:num_neighbors] = True  # Vizinhos reais

        # Concatena a máscara à matriz de observação
        observation = np.hstack((observation_matrix, mask.reshape(-1, 1).astype(np.float32)))

        # Obtém o ground truth do SensorManager
        ground_truth = self.sensor_manager.get_ground_truth()

        return observation, ground_truth











    
    def _prepare_features(
        self, neighbors: pd.DataFrame, target_row: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Prepara a matriz de observação (u_matrix), a máscara e o ground truth.

        Args:
            neighbors (pd.DataFrame): DataFrame com os vizinhos selecionados.
            target_row (pd.Series): Linha de dados do sensor alvo.

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: u_matrix, mask e ground_truth.
                - u_matrix: (n_neighbors_max, 4)
                - mask: (n_neighbors_max,)
                - ground_truth: float
        """
        u_matrix = np.zeros((self.n_neighbors_max, 4), dtype=np.float32)
        mask = np.zeros(self.n_neighbors_max, dtype=bool)

        for i in range(self.n_neighbors_max):
            if i < len(neighbors):
                neighbor = neighbors.iloc[i]
                u_matrix[i, :] = [
                    neighbor["lat"] - target_row["lat"],
                    neighbor["lon"] - target_row["lon"],
                    neighbor["t"] - target_row["t"],
                    neighbor["y"],
                ]
                mask[i] = True
            else:
                # Preencher com zeros para vizinhos inexistentes
                u_matrix[i, :] = [0.0, 0.0, 0.0, 0.0]
                mask[i] = False

        ground_truth = target_row["y"]

        return u_matrix, mask, ground_truth

    def close(self):
        """
        Fecha o ambiente.
        """
        pass
