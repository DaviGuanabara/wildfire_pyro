import os
import torch
from wildfire_pyro.environments.fixed_sensor_environment import FixedSensorEnvironment
from wildfire_pyro.factories.model_factory import (
    create_deep_set_attention_net as create_model,
)


print("Learning Example está em construção")


def get_path(file_name):
    # Caminho do arquivo de teste
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join("data", "synthetic", "fixed_sensor", file_name)
    data_path = os.path.join(SCRIPT_DIR, relative_path)
    return data_path

train_data = get_path("fixed_train.csv")
test_data = get_path("fixed_test.csv")

# Configurações do ambiente
max_steps = 200000
n_neighbors_min = 2
n_neighbors_max = 5

# Inicializa o ambiente
# TODO: adicioanr uma flag para que eu deixe o número de steps com o tamanho da base de dados.
train_environment = FixedSensorEnvironment(
    data_path=train_data,
    max_steps=max_steps,
    n_neighbors_min=n_neighbors_min,
    n_neighbors_max=n_neighbors_max,
)


test_environment = FixedSensorEnvironment(
    data_path=test_data,
    max_steps=max_steps,
    n_neighbors_min=n_neighbors_min,
    n_neighbors_max=n_neighbors_max,
)



# Configurações para a rede neural e o treinamento
parameters = {
    "lr": 0.1,  # Taxa de aprendizado
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Device para treinamento
    "dropout_prob": 0.2,  # Probabilidade de dropout na rede neural
    "hidden": 64,  # Neurônios na camada oculta
    "batch_size": 128,  # Tamanho do batch para treinamento
}

#TODO: BUSCAR UM NOME MELHOR PARA O CREATE_MODEL, PQ AFINAL, ELE GERA O MODELO DENTRO DE UM WRAPPER.
# Função que cria o modelo e o gerenciador de aprendizado
deep_set = create_model(train_environment, parameters)


# Definição do número total de passos de treinamento e passos por rollout
total_steps = 10000

# TODO: Adicionar SEED no reset.
seed = 0
train_environment.reset(seed)
# Executa o processo de aprendizagem
# TODO: ADICIONAR VALIDAÇÃO PARA ACOMPANHAR TREINAMENTO
# TODO: ADICIONAR CALLBACKS DO SB3
deep_set.learn(total_steps)

train_environment.close()
print("aprendizagem concluída")

# Teste de inferência após o treinamento
# TODO: O Treinamento, seguindo o arquigo, segue um bootstrap de 20 conjuntos de vizinhos (ou seja, 20 observações.)
observation, info = test_environment.reset()
for _ in range(5):
    
    
    action, _ = deep_set.predict(observation)
    error = action.item() - info['ground_truth']
    print(f">> Action taken: {action.item():.4f}, Error: {error:.4f}")
    
    observation, reward, terminated, truncated, info = test_environment.step(action)
    print("Ação prevista:", action, "Ground truth:", info["ground_truth"])

# Não se esqueça de fechar o ambiente para liberar recursos
test_environment.close()




