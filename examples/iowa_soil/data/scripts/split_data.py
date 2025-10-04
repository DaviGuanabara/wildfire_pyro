import pandas as pd
from sklearn.model_selection import train_test_split

# Caminho do arquivo original
INPUT_FILE = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\wildfire_pyro\\examples\\iowa_soil\\data\\processed\\tidy_isusm_stations.csv"

# Carregar o dataset
df = pd.read_csv(INPUT_FILE)

# Dividir em treino (70%) e temp (30%)
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, shuffle=True)

# Dividir temp em validação (15%) e teste (15%)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, shuffle=True)

# Salvar os arquivos
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Tamanho original: {len(df)} linhas")
print(f"Treinamento: {len(train_df)} linhas")
print(f"Validação: {len(val_df)} linhas")
print(f"Teste: {len(test_df)} linhas")
