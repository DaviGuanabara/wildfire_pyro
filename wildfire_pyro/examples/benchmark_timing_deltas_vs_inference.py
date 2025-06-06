import time
import torch
import numpy as np
from wildfire_pyro.environments.sensor_environment import SensorEnvironment
from wildfire_pyro.factories.learner_factory import create_deep_set_learner
from wildfire_pyro.common.seed_manager import configure_seed_manager, get_seed

# ==========================================
# Configuração
# ==========================================
global_seed = 42
configure_seed_manager(global_seed)

# Substitua pelo caminho correto
train_data = "data/synthetic/fixed_sensor/fixed_train.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"

env = SensorEnvironment(
    data_path=train_data,
    max_steps=1,
    n_neighbors_min=5,
    n_neighbors_max=30,
    verbose=False
)

# Modelo (carregado como no projeto)
model_parameters = {
    "lr": 1e-3,
    "dropout_prob": 0.2,
    "hidden": 64,
    "batch_size": 128,
}
runtime_parameters = {
    "device": device,
    "verbose": 0,
    "seed": get_seed("benchmark_run")
}
logging_parameters = {"log_path": None, "format_strings": []}

learner = create_deep_set_learner(
    env,
    model_parameters=model_parameters,
    logging_parameters=logging_parameters,
    runtime_parameters=runtime_parameters,
)

model = learner.neural_network.to(device)
model.eval()

# ==========================================
# Parte 1 - Medir tempo de deltas
# ==========================================
start = time.time()
obs, gt = env.get_bootstrap_observations(n_bootstrap=4)
delta_time = time.time() - start
print(f"[⏱️ DELTA] Tempo para gerar deltas: {delta_time:.4f} segundos")

# ==========================================
# Parte 2 - Medir tempo de inferência
# ==========================================
obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

start = time.time()
with torch.no_grad():
    output = model(obs_tensor)
inference_time = time.time() - start
print(
    f"[⏱️ INFERÊNCIA] Tempo de predição do modelo: {inference_time:.4f} segundos")

# ==========================================
# Resultado
# ==========================================
print(f"\n🔍 Conclusão:")
print(f"  > Tempo gasto em Deltas: {delta_time:.4f}s")
print(f"  > Tempo gasto em Inferência: {inference_time:.4f}s")
print(f"  > Tamanho da entrada: {obs_tensor.shape}")
