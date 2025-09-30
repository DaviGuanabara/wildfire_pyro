import os
import pandas as pd

# Arquivos de entrada
file_data = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\wildfire_pyro\\examples\\iowa_soil\\data\\processed\\tidy_isusm.csv"      # arquivo 1
file_meta = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\wildfire_pyro\\examples\\iowa_soil\\data\\raw\\stations.xlsx"  # arquivo 2


print("CSV existe?", os.path.exists(file_data))
print("XLSX existe?", os.path.exists(file_meta))

# Carregar
df_data = pd.read_csv(file_data)
# arquivo 2 parece ser tab-delimited
df_meta = pd.read_excel(file_meta)

# Juntar pelo ID/station
# Padronizar chaves
df_data["station"] = df_data["station"].astype(str).str.strip().str.upper()
df_meta["ID"] = df_meta["ID"].astype(str).str.strip().str.upper()

df_data["valid"] = pd.to_datetime(df_data["valid"])
df_data["valid"] = df_data["valid"].map(lambda d: d.toordinal())
df_data["valid"] = df_data["valid"].astype(int)

# Verificar diferenças
ids_data = set(df_data["station"])
ids_meta = set(df_meta["ID"])
print("IDs só no CSV:", ids_data - ids_meta)
print("IDs só no Excel:", ids_meta - ids_data)

# Merge
df_merged = pd.merge(
    df_data,
    df_meta,
    left_on="station",
    right_on="ID",
    how="inner"
)

# Salvar
df_merged.to_csv("estacoes_completas.csv", index=False)

print(
    f"Arquivo final salvo com {len(df_merged)} linhas e {len(df_merged.columns)} colunas")
