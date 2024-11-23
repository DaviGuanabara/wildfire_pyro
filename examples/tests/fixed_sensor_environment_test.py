import sys
import os

# Adicione o diretório raiz ao sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, root_dir)

import unittest
import numpy as np
from wildfire_private.examples.fixed_sensor_environment import Fixed_Sensor_Environment

class TestFixedSensorEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Configuração inicial para carregar o dataset e criar o ambiente.
        """
        cls.data_path = "data/synthetic/fixed_sensor/fixed_train.csv"  # Substitua pelo caminho correto
        cls.env = Fixed_Sensor_Environment(cls.data_path)

    def test_initialization(self):
        """
        Testa se o ambiente é inicializado corretamente.
        """
        self.assertIsInstance(self.env, Fixed_Sensor_Environment)
        self.assertGreater(len(self.env.data), 0, "Dataset não carregado corretamente.")

    def test_reset(self):
        """
        Testa o método reset para verificar se retorna observações válidas.
        """
        observation, info = self.env.reset()
        self.assertIsInstance(observation, tuple, "Observação deve ser uma tupla (u_matrix, mask).")
        self.assertIn("ground_truth", info, "As informações devem conter 'ground_truth'.")
        self.assertEqual(len(observation[0]), 4, "A matriz u_matrix deve ter 4 dimensões (features).")
        self.assertEqual(len(observation[1]), self.env.n_neighbors_max, "O mask deve ter tamanho igual ao número máximo de vizinhos.")

    def test_step(self):
        """
        Testa o método step para garantir que ele retorna os valores esperados.
        """
        self.env.reset()
        action = np.array([0.5])  # Ação fictícia
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.assertIsInstance(observation, tuple, "Observação após step deve ser uma tupla.")
        self.assertIsInstance(reward, float, "Recompensa deve ser um número float.")
        self.assertIsInstance(terminated, bool, "Terminado deve ser um valor booleano.")
        self.assertIn("ground_truth", info, "As informações devem conter 'ground_truth'.")

    def test_observation_limits(self):
        """
        Verifica se os valores da observação estão dentro dos limites esperados.
        """
        observation, _ = self.env.reset()
        u_matrix, _ = observation

        self.assertTrue((u_matrix[0] >= -self.env.lat_max).all() and (u_matrix[0] <= self.env.lat_max).all(),
                        "Os valores de Δlat estão fora dos limites.")
        self.assertTrue((u_matrix[1] >= -self.env.lon_max).all() and (u_matrix[1] <= self.env.lon_max).all(),
                        "Os valores de Δlon estão fora dos limites.")
        self.assertTrue((u_matrix[2] >= -3 * np.pi).all() and (u_matrix[2] <= 0).all(),
                        "Os valores de Δt estão fora dos limites.")

    def test_termination(self):
        """
        Testa se o ambiente termina corretamente ao atingir o número máximo de etapas.
        """
        self.env.reset()
        for _ in range(self.env.max_steps):
            _, _, terminated, _, _ = self.env.step(np.array([0.5]))
        self.assertTrue(terminated, "O ambiente não termina corretamente após atingir o máximo de etapas.")

    @classmethod
    def tearDownClass(cls):
        """
        Finaliza o ambiente após todos os testes.
        """
        cls.env.close()


if __name__ == "__main__":
    unittest.main()
