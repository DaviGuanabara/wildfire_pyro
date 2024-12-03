import os
import numpy as np
import pytest
from wildfire_pyro.environments.fixed_sensor_environment import FixedSensorEnvironment


@pytest.fixture(scope="module")
def fixed_sensor_env():
    """
    Fixture para criar e configurar o ambiente FixedSensorEnvironment.
    """
    data_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data/synthetic/fixed_sensor/fixed.csv",
        )
    )
    env = FixedSensorEnvironment(data_path)
    yield env
    env.close()


def test_initialization(fixed_sensor_env):
    """
    Testa se o ambiente é inicializado corretamente.
    """
    env = fixed_sensor_env
    assert isinstance(
        env, FixedSensorEnvironment
    ), "O ambiente não foi inicializado corretamente."
    assert len(env.sensor_manager.data) > 0, "Dataset não carregado corretamente."

"""
def test_reset(fixed_sensor_env):
    
    #Testa o método reset para verificar se retorna observações válidas.
    
    env = fixed_sensor_env
    observation, info = env.reset()
    assert isinstance(
        observation, tuple
    ), "Observação deve ser uma tupla (u_matrix, mask)."
    assert "ground_truth" in info, "As informações devem conter 'ground_truth'."
    assert (
        len(observation[0]) == 4
    ), "A matriz u_matrix deve ter 4 dimensões (features)."
    assert (
        len(observation[1]) == env.n_neighbors_max
    ), "O mask deve ter tamanho igual ao número máximo de vizinhos."
"""

import numpy as np


def test_step(fixed_sensor_env):
    """
    Testa o método step para garantir que ele retorna os valores esperados.
    """
    env: FixedSensorEnvironment = fixed_sensor_env
    env.reset()
    action = np.array([0.5])  # Ação fictícia
    observation, reward, terminated, truncated, info = env.step(action)

    # Suponha que 'observation' seja a saída do ambiente
    # observation.shape = (n_neighbors_max, 5)

    u_matrix: np.ndarray = observation[:, :4]  # Primeiras 4 colunas
    mask: np.ndarray = observation[:, 4]  # Última coluna

    #print("observation")
    #print(observation)
    
    print("u_matrix shape:", u_matrix.shape)
    print(u_matrix)
    print("mask shape:", mask.shape)
    print(mask)

    print("n neighbors max:", env.n_neighbors_max)

    assert u_matrix.shape == (
        env.n_neighbors_max,
        4,
    ), "A matriz u_matrix deve ter o shape (n_neighbors_max, 4)."
    assert mask.shape == (
        env.n_neighbors_max,
    ), "O mask deve ter shape (n_neighbors_max,)."

    # assert isinstance(u_matrix, tuple), "Observação após step deve ser uma tupla."
    # assert isinstance(reward, float), "Recompensa deve ser um número float."
    assert isinstance(terminated, bool), "Terminado deve ser um valor booleano."
    assert "ground_truth" in info, "As informações devem conter 'ground_truth'."


def test_termination(fixed_sensor_env):
    """
    Testa se o ambiente termina corretamente ao atingir o número máximo de etapas.
    """
    env = fixed_sensor_env
    env.reset()
    for _ in range(env.max_steps):
        _, _, terminated, _, _ = env.step(np.array([0.5]))
    assert (
        terminated
    ), "O ambiente não termina corretamente após atingir o máximo de etapas."
