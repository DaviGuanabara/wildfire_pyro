from wildfire_pyro.environments.sensor.sensor_environment import SensorEnvironment
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
environment = SensorEnvironment(
    data_path=data_path,
    max_steps=max_steps,
    n_neighbors_min=n_neighbors_min,
    n_neighbors_max=n_neighbors_max,
)


print("\n=== Starting new episode  ===")
observation, info = environment.reset()

for step in range(1, 5):
    

    # Agent's action	
    action = np.array([np.random.uniform(-1, 1)])
    error = action.item() - info['ground_truth']
    print(f">> Action taken: {action.item():.4f}, Error: {error:.4f}")

    
    observation, reward, terminated, truncated, info = environment.step(action)
    
    print("\n")
    print(f"--- Step {step}/{max_steps} ---")
    print(f">> Sensor under evaluation (sensor id = {info['sensor']['sensor_id']}) - Latitude: {info['sensor']['lat']:.4f}, Longitude: {info['sensor']['lon']:.4f}")
    print(f">> Ground Truth of sensor {info['sensor']['sensor_id']} no tempo {info['sensor']['t']:.4f}: {info['ground_truth']:.4f}")
    

    for idx, row in enumerate(observation):
        rounded_row = np.around(row, decimals=4)
        is_valid = "Valid" if np.any(row) else "Invalid"
        print(f"    Delta Neighbor  {idx + 1} ({is_valid}): {rounded_row}")

    if terminated:
        print("\n The episode has ended.")
        break

environment.close()
