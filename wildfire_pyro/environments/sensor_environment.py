import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional
from gymnasium import spaces, Env

from wildfire_pyro.environments.components.sensor_manager import SensorManager
from wildfire_pyro.environments.base_environment import BaseEnvironment

# Configuração básica do logging
logging.basicConfig(level=logging.INFO)


class SensorEnvironment(BaseEnvironment):
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
        super(SensorEnvironment, self).__init__()

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

        return observation, {
            "ground_truth": self.ground_truth,
            "sensor": self._sensor_info(),
        }

    # TODO: a ação deve mesmo ser um array ?
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

        # TODO: Adicionar o baseline como retorno no info ?
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
        sensor_data = self.sensor_manager.cache["data_from_current_sensor"].iloc[
            current_index
        ]

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
        deltas = self.sensor_manager.get_neighbors_deltas(
            self.n_neighbors_max, self.n_neighbors_min
        )

        # Criação da matriz de observação (n_neighbors_max, 4)
        # TODO: Esse hardcoded 4 é o tamanho do vetor de deltas. Deveria ser um parâmetro
        # retirado da própria classe SensorManager, da base de dados.
        observation_matrix = np.zeros((self.n_neighbors_max, 4), dtype=np.float32)
        mask = np.zeros(self.n_neighbors_max, dtype=bool)

        # Preenche a matriz com os deltas existentes
        num_neighbors = len(deltas)
        observation_matrix[:num_neighbors, :] = deltas.values
        mask[:num_neighbors] = True  # Vizinhos reais

        # Concatena a máscara à matriz de observação
        observation = np.hstack(
            (observation_matrix, mask.reshape(-1, 1).astype(np.float32))
        )

        # Obtém o ground truth do SensorManager
        ground_truth = self.sensor_manager.get_ground_truth()

        return observation, ground_truth

    def close(self):
        """
        Fecha o ambiente.
        """
        pass

    def get_bootstrap_observations(self, n_bootstrap: int) -> Tuple[list, float]:
        """
        Generates a list of observations using bootstrap neighbor deltas and returns a single ground truth value.

        For each bootstrap sample, the SensorManager's get_bootstrap_neighbors_deltas method is used to
        obtain a delta DataFrame. An observation matrix is built from each delta DataFrame by:
        - Creating a feature matrix of shape (n_neighbors_max, 4), where available delta values are placed
            in the upper rows.
        - Creating a boolean mask of length n_neighbors_max (True for rows with data, False for padded rows).
        - Horizontally stacking the feature matrix and the mask (converted to float) to produce an observation
            of shape (n_neighbors_max, 5).

        Since the target sensor is locked, the ground truth is computed once and is identical for all samples.

        Args:
            n_bootstrap (int): Number of bootstrap samples to generate.

        Returns:
            Tuple[list, float]:
                - A list of observation matrices (each with shape (n_neighbors_max, 5)).
                - A single ground truth value from the target sensor.
        """
        # Obtain the bootstrap neighbor deltas and the ground truth from the sensor manager.
        bootstrap_deltas, ground_truth = (
            self.sensor_manager.get_bootstrap_neighbors_deltas(
                n_bootstrap=n_bootstrap,
                n_neighbors_max=self.n_neighbors_max,
                n_neighbors_min=self.n_neighbors_min,
                time_window=-1,  # Adjust if needed.
                distance_window=-1,  # Currently unused.
                force_recompute=False,
            )
        )

        observations = []

        # For each bootstrap sample, build the observation.
        for deltas in bootstrap_deltas:
            # Create an empty feature matrix and a mask.
            observation_matrix = np.zeros((self.n_neighbors_max, 4), dtype=np.float32)
            mask = np.zeros(self.n_neighbors_max, dtype=bool)

            # Get the number of available neighbors in this sample.
            num_neighbors = len(deltas)

            # Fill in the feature matrix with the available delta values.
            if num_neighbors > 0:
                observation_matrix[:num_neighbors, :] = deltas.values

            # Mark the positions corresponding to real neighbors.
            mask[:num_neighbors] = True

            # Stack the feature matrix and mask to form the final observation.
            # The mask is converted to float (1.0 for True, 0.0 for False).
            observation = np.hstack(
                (observation_matrix, mask.reshape(-1, 1).astype(np.float32))
            )
            observations.append(observation)

        # TODO: Trocar ground_truth por info, e a adicionar no info o ground_truth e o baseline ?
        return observations, ground_truth

    def baseline(self):
        """
        Returns a baseline estimation using the mean and standard deviation of the
        neighbors' 'y' values from the most recent bootstrap.

        This method is optional and not required for the system to operate.
        However, if implemented, evaluation callbacks may use this as a reference
        to compare the learner's performance during training or inference.

        Returns:
            Tuple[float, float]: (mean_y, std_y) if available; otherwise, None.
        """
        bootstrap_deltas, _ = self.sensor_manager.get_bootstrap_neighbors_deltas(
            force_recompute=False
        )

        mean_y = np.mean([df["y"].mean() for df in bootstrap_deltas])
        std_y = np.std([df["y"].mean() for df in bootstrap_deltas])

        return mean_y, std_y
