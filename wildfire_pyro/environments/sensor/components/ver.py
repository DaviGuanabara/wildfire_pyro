import pandas as pd
import numpy as np

# Caminho para o arquivo CSV
data_path = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire_pyro\\tests\\data\\synthetic\\fixed_sensor\\fixed.csv"

# Carregar os dados do CSV
data = pd.read_csv(data_path)

# Adicionar uma coluna 'sensor_id' para identificar sensores únicos
data["sensor_id"] = data.groupby(["lat", "lon"]).ngroup()

# Obter os IDs únicos dos sensores
sensors = data["sensor_id"].unique()

# Escolher aleatoriamente o número de vizinhos (entre 3 e 5)
num_neighbors = np.random.randint(3, 6)

# Selecionar aleatoriamente os sensores para serem os vizinhos
neighbor_sensors = np.random.choice(sensors, size=num_neighbors, replace=False)

# Criar uma lista para armazenar os vizinhos escolhidos
selected_neighbors = []

# Iterar sobre os sensores selecionados e escolher aleatoriamente uma
# leitura de cada um
for sensor_id in neighbor_sensors:
    # Filtrar os dados do sensor atual
    sensor_data = data[data["sensor_id"] == sensor_id]

    # Escolher aleatoriamente um índice de tempo dentro dos dados disponíveis
    random_index = np.random.choice(sensor_data.index)
    selected_neighbors.append(sensor_data.loc[random_index])

# Converter a lista de vizinhos escolhidos em um DataFrame
neighbors_df = pd.DataFrame(selected_neighbors)

# Exibir os resultados
print(f"Número total de sensores: {len(sensors)}")
print(f"Número de vizinhos escolhidos: {num_neighbors}")
print(f"Vizinhos escolhidos:\n{neighbors_df}")
