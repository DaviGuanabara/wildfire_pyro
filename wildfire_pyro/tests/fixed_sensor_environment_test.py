import os
import numpy as np
import pytest
from wildfire_pyro.environments.sensor_environment import SensorEnvironment


@pytest.fixture(scope="module")
def fixed_sensor_env():
    """
    Fixture para criar e configurar o ambiente SensorEnvironment.
    """
    data_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data/synthetic/fixed_sensor/fixed.csv",
        )
    )
    env = SensorEnvironment(data_path)
    yield env
    env.close()

