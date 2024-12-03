import os
import pandas as pd
import numpy as np
import logging
import pytest
from wildfire_pyro.environments.components.sensor_manager import SensorManager

logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope="module")
def sensor_manager():
    """
    Fixture para criar e configurar o SensorManager.
    """
    data_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data/synthetic/fixed_sensor/fixed.csv",  # Substitua pelo caminho correto do seu dataset
        )
    )
    sm = SensorManager(data_path)
    sm.set_random_sensor()  # Seleciona um sensor inicial
    return sm


def test_sensor_manager_initialization(sensor_manager: SensorManager):
    """
    Testa a inicialização do SensorManager.
    """
    sm = sensor_manager
    assert (
        sm.current_sensor is not None
    ), "Nenhum sensor foi selecionado após update_sensor()."
    assert sm.current_time_index >= 0, "O índice de tempo inicial deve ser maior ou igual a 0."

def test_get_neighbors(sensor_manager: SensorManager):
    """
    Testa a função get_neighbors.
    """

    min_neighbors = 1
    max_neighbors = 5

    sm = sensor_manager
    neighbors = sm.get_neighbors(
        n_neighbors_min=min_neighbors,
        n_neighbors_max=max_neighbors,
        time_window=2,
    )

    print("Vizinhos:", len(neighbors))
    print(neighbors)
    assert isinstance(
        neighbors, pd.DataFrame
    ), "A saída de get_neighbors deve ser um DataFrame."
    assert (
        len(neighbors) >= min_neighbors
    ), f"O número de vizinhos é menor que o mínimo especificado ({min_neighbors})."
    assert (
        len(neighbors) <= max_neighbors
    ), f"O número de vizinhos é maior que o máximo especificado ({max_neighbors})."


def test_sensor_reading(sensor_manager):
    """
    Testa a função get_reading.
    """
    sm = sensor_manager
    reading = sm.get_reading()

    assert isinstance(reading, pd.Series), "A leitura deve ser um pandas Series."
    assert (
        "t" in reading and "lat" in reading and "lon" in reading and "y" in reading
    ), "A leitura deve conter as colunas 't', 'lat', 'lon', e 'y'."


def test_neighbors_limits(sensor_manager: SensorManager):
    """
    Testa os limites de vizinhos retornados pela função get_neighbors.
    """
    sm = sensor_manager
    sm.set_random_time()
    
    neighbors = sm.get_neighbors(
        time_window=2,
        n_neighbors_min=3,
        n_neighbors_max=5,
    )

    assert len(neighbors) >= 3, "Número de vizinhos é menor que o mínimo permitido (3)."
    assert len(neighbors) <= 5, "Número de vizinhos é maior que o máximo permitido (5)."


def test_sensor_random_time(sensor_manager: SensorManager):
    """
    Testa a função set_random_time.
    """
    sm = sensor_manager
    random_time_index = sm.set_random_time()
    assert (
        0 <= random_time_index < len(sm.data_from_current_sensor)
    ), "O índice de tempo aleatório está fora do intervalo permitido."


def test_sensor_position(sensor_manager):
    """
    Testa a função get_position.
    """
    sm = sensor_manager
    position = sm.get_position()

    assert isinstance(position, tuple), "A posição deve ser retornada como uma tupla."
    assert (
        len(position) == 2
    ), "A posição deve conter dois valores: latitude e longitude."
