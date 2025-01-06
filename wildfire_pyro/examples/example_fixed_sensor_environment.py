from ..environments.fixed_sensor_environment import Fixed_Sensor_Environment
import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


# Configurações do ambiente
data_path = "path/to/your/dataset.csv"
max_steps = 20
n_neighbors_min = 2
n_neighbors_max = 5

# Inicializa o ambiente
environment = Fixed_Sensor_Environment(
    data_path=data_path,
    max_steps=max_steps,
    n_neighbors_min=n_neighbors_min,
    n_neighbors_max=n_neighbors_max,
)

# Reinicia o ambiente para começar um novo episódio
observation, info = environment.reset()
print("Initial Observation:", observation)
print("Initial Ground Truth:", info["ground_truth"])

# Simula uma interação de agente com o ambiente
for step in range(max_steps):
    print(f"\n--- Step {step + 1} ---")

    # Escolha de ação (simulada aleatoriamente aqui)
    action = np.array([np.random.uniform(-1, 1)])
    print("Action:", action)

    # Executa o passo no ambiente
    observation, reward, terminated, truncated, info = environment.step(action)

    # Exibe os resultados do passo
    print("Observation:", observation)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    if terminated:
        print("Episode terminated.")
        break

# Encerrar o ambiente (se necessário para liberar recursos)
environment.close()
