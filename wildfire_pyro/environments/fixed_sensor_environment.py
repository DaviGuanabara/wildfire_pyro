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

    # TODO: O SEED DEVE SER AJUSTADO CORRETAMENTE.
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
        self.sensor_manager.set_random_sensor()

        observation, self.ground_truth = self._generate_observation()

        return observation, {"ground_truth": self.ground_truth}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executa um passo no ambiente.

        Args:
            action (np.ndarray): A ação tomada pelo agente.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: Observação, recompensa, terminated, truncated, informações.
        """

        # a ação foi gerada dada a última leitura do sensor, assim a comparação com o ground_truth
        # deve ser feita antes da atualização do sensor
        reward = self._calculate_reward(action, self.ground_truth)

        self.current_step += 1
        self.sensor_manager.set_random_sensor()

        observation, ground_truth = self._generate_observation()

        self.ground_truth = ground_truth

        terminated = self._is_terminated()
        truncated = False  # Pode ser ajustado conforme a lógica do ambiente

        return (
            observation,
            reward,
            terminated,
            truncated,
            {"ground_truth": ground_truth, "sensor": self._sensor_info()},
        )

    def _sensor_info(self) -> dict:

        lat, lon = self.sensor_manager.get_current_sensor_position()
        sensor_id = self.sensor_manager.current_sensor
        sensor_t = self.sensor_manager.get_current_sensor_time()

        sensor_info = {
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "t": round(sensor_t, 4),
            "sensor_id": sensor_id,
        }

        return sensor_info

    def _calculate_reward(self, action: np.ndarray, ground_truth: float) -> float:
        """
        Calcula a recompensa para o passo atual.

        Args:
            action (np.ndarray): Ação tomada pelo agente.
            ground_truth (float): Valor de ground truth.

        Returns:
            float: Recompensa calculada.
        """
        error = np.abs(action[0] - ground_truth)
        reward = -error
        logging.debug(f"Recompensa calculada: {reward} (Erro: {error})")
        return float(reward)

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
        target_row = self.sensor_manager.get_reading()
        logging.debug(f"target row: {target_row}")

        #TODO: EU TÔ PEGANDO OS VIZINHOS E JÁ JOGANDO COMO OBSERVAÇÃO.
        #PORÉM, ERA PARA ESTAR GERANDO OS DELTAS DOS VIZINHOS, E AÍ JOGAR NA OBSERVAÇÃO.
        
        neighbors = self.sensor_manager.get_neighbors(
            n_neighbors_max=self.n_neighbors_max,
            n_neighbors_min=self.n_neighbors_min,
            time_window=3 * np.pi,
            distance_window=-1,
        )

        u_matrix, mask, ground_truth = self._prepare_features(neighbors, target_row)

        # Concatenar u_matrix e mask horizontalmente
        # u_matrix: (n_neighbors_max, 4)
        # mask: (n_neighbors_max,)
        # Reshape mask para (n_neighbors_max, 1) antes da concatenação
        observation = np.hstack(
            (u_matrix, mask.reshape(-1, 1).astype(np.float32))
        )  # (n_neighbors_max, 5)

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
