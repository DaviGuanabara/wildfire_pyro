import pandas as pd
from sklearn.model_selection import train_test_split

# Caminho do arquivo original
# INPUT_FILE = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\examples\\iowa_soil\\data\\daily\\processed\\dataset_with_baseline.csv"
INPUT_FILE = "/Users/Davi/Documents/GitHub/wildfire_workspace/wildfire/experiments/iowa_soil/data/daily/processed/dataset_with_baseline.csv"
# Carregar o dataset
df = pd.read_csv(INPUT_FILE)

# Dividir em treino (80%) e teste (20%)
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)


# Salvar os arquivos
train_df.to_csv("train-final.csv", index=False)
test_df.to_csv("test-final.csv", index=False)

print(f"Tamanho original: {len(df)} linhas")
print(f"Treinamento: {len(train_df)} linhas")
print(f"Teste: {len(test_df)} linhas")
