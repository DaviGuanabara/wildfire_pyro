import torch
from gymnasium import make as gym_make

from wildfire_pyro.factories.model_factory import create_deep_set_attention_net as create_model
from wildfire_pyro.wrappers.learning_manager import LearningManager
from wildfire_pyro.environments.fixed_sensor_environment import FixedSensorEnvironment
import numpy as np
import sys
import os


from pathlib import Path


#Caminho do arquivo de teste
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join("data", "synthetic", "fixed_sensor", "fixed_test.csv")
data_path = os.path.join(SCRIPT_DIR, relative_path)

# Configurações do ambiente
max_steps = 20
n_neighbors_min = 2
n_neighbors_max = 5

# Inicializa o ambiente
environment = FixedSensorEnvironment(
    data_path=data_path,
    max_steps=max_steps,
    n_neighbors_min=n_neighbors_min,
    n_neighbors_max=n_neighbors_max,
)



# Configurações para a rede neural e o treinamento
parameters = {
    "lr": 0.01,  # Taxa de aprendizado
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Device para treinamento
    "dropout_prob": 0.2,  # Probabilidade de dropout na rede neural
    "hidden": 64,  # Neurônios na camada oculta
    "batch_size": 128  # Tamanho do batch para treinamento
}

# Função que cria o modelo e o gerenciador de aprendizado
deep_set = create_model(environment, parameters)




# Definição do número total de passos de treinamento e passos por rollout
total_steps = 1000
rollout_steps = 100

# Executa o processo de aprendizagem
deep_set.learn(total_steps, rollout_steps)

# Teste de inferência após o treinamento
observation = environment.reset()
action, _ = deep_set.predict(observation)
print("Ação prevista:", action)

# Não se esqueça de fechar o ambiente para liberar recursos
environment.close()
