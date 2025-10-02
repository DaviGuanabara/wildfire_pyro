import os
import numpy as np
from wildfire_pyro.environments.sensor.sensor_environment import SensorEnvironment

# Set up the path to the test data file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join("data", "synthetic", "fixed_sensor", "fixed_test.csv")
data_path = os.path.join(SCRIPT_DIR, relative_path)

# Environment configurations
max_steps = 20
n_neighbors_min = 2
n_neighbors_max = 5

# Initialize the environment
environment = SensorEnvironment(
    data_path=data_path,
    max_steps=max_steps,
    n_neighbors_min=n_neighbors_min,
    n_neighbors_max=n_neighbors_max,
)

print("\n=== Starting new episode with Bootstrap Observations ===")
observation, info = environment.reset()
print(f">> Initial Sensor (ID: {info['sensor']['sensor_id']})")
print(f"   Latitude: {info['sensor']['lat']:.4f}, Longitude: {info['sensor']['lon']:.4f}")
print(f">> Ground Truth: {info['ground_truth']:.4f}")

# Define the number of bootstrap samples you want
n_bootstrap = 20

# Loop for 3 steps: at each step, change the sensor via environment.step and then get bootstrap observations.
for step in range(1, 4):
    # Generate a random action (the actual value may be irrelevant if action isn't used to select the sensor)
    action = np.array([np.random.uniform(-1, 1)])
    
    # Advance the environment (this calls sensor_manager.step to select a new sensor)
    observation, reward, terminated, truncated, info = environment.step(action)
    
    print(f"\n--- Step {step} ---")
    print(f">> Sensor under evaluation (ID: {info['sensor']['sensor_id']})")
    print(f"   Latitude: {info['sensor']['lat']:.4f}, Longitude: {info['sensor']['lon']:.4f}")
    print(f">> Ground Truth: {info['ground_truth']:.4f}")
    
    # Now get the bootstrap observations for the current (locked) sensor.
    bootstrap_obs, ground_truth = environment.get_bootstrap_observations(n_bootstrap)
    print(f"\nBootstrap observations (common Ground Truth: {ground_truth:.4f}):")
    
    # Iterate over each bootstrap sample
    for idx, obs in enumerate(bootstrap_obs):
        print(f"\n  Bootstrap Sample {idx + 1}:")
        # Each 'obs' is an observation matrix of shape (n_neighbors_max, 5)
        # The first 4 columns are the features (delta values) and the 5th column is the mask.
        for neighbor_idx, row in enumerate(obs):
            rounded_row = np.around(row, decimals=4)
            # Check if the row contains valid data (at least one nonzero in the first 4 columns)
            is_valid = "Valid" if np.any(row[:-1]) else "Invalid"
            print(f"    Delta Neighbor {neighbor_idx + 1} ({is_valid}): {rounded_row}")
    
    if terminated:
        print("\nEpisode terminated.")
        break

environment.close()
