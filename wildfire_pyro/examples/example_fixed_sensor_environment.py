from wildfire_pyro.environments.fixed_sensor_environment import FixedSensorEnvironment
import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


# Configurações do ambiente
data_path = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_project\\workspace\\wildfire_pyro\\examples\\data\\synthetic\\fixed_sensor\\fixed_test.csv"
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

print(environment.sensor_manager.sensors)
# Reinicia o ambiente para começar um novo episódio
print("\n=== Iniciando novo episódio ===")
observation, info = environment.reset()
print(">> Observação inicial:")
print(observation)
print(">> Ground Truth inicial:", info["ground_truth"])

# Simula uma interação de agente com o ambiente
for step in range(1, max_steps + 1):
    print(f"\n--- Passo {step}/{max_steps} ---")

    # Escolha de ação (simulada aleatoriamente aqui)
    action = np.array([np.random.uniform(-1, 1)])
    print(f">> Ação escolhida: {action[0]:.4f}")

    # Executa o passo no ambiente
    observation, reward, terminated, truncated, info = environment.step(action)

    # Exibe os resultados formatados
    print("\n>> Observação recebida (vizinhos selecionados):")
    for idx, row in enumerate(observation):
        if np.any(row):  # Exibe apenas as linhas preenchidas
            print(f"  Vizinho {idx + 1}: {row}")
    
    print(f">> Recompensa recebida: {reward:.4f}")
    print(f">> Episódio terminado? {'Sim' if terminated else 'Não'}")
    print(f">> Episódio truncado? {'Sim' if truncated else 'Não'}")
    print(f">> Ground Truth atual: {info['ground_truth']:.4f}")

    print(environment.sensor_manager.current_sensor)

    if terminated:
        print("\n[INFO] O episódio foi encerrado.")
        break

print("\n=== Simulação concluída ===")
environment.close()