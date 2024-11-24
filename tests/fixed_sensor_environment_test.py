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


def test_reset(fixed_sensor_env):
    """
    Testa o método reset para verificar se retorna observações válidas.
    """
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


def test_step(fixed_sensor_env):
    """
    Testa o método step para garantir que ele retorna os valores esperados.
    """
    env = fixed_sensor_env
    env.reset()
    action = np.array([0.5])  # Ação fictícia
    observation, reward, terminated, truncated, info = env.step(action)

    assert isinstance(observation, tuple), "Observação após step deve ser uma tupla."
    #assert isinstance(reward, float), "Recompensa deve ser um número float."
    assert isinstance(terminated, bool), "Terminado deve ser um valor booleano."
    assert "ground_truth" in info, "As informações devem conter 'ground_truth'."

"""
    Verifica se os valores da observação estão dentro dos limites esperados.
"""
"""
def test_observation_limits(fixed_sensor_env):
    
    env = fixed_sensor_env
    observation, _ = env.reset()
    u_matrix, _ = observation

    assert (u_matrix[0] >= -env.lat_max).all() and (
        u_matrix[0] <= env.lat_max
    ).all(), "Os valores de Δlat estão fora dos limites."
    assert (u_matrix[1] >= -env.lon_max).all() and (
        u_matrix[1] <= env.lon_max
    ).all(), "Os valores de Δlon estão fora dos limites."
    assert (u_matrix[2] >= -3 * np.pi).all() and (
        u_matrix[2] <= 0
    ).all(), "Os valores de Δt estão fora dos limites."
"""

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



