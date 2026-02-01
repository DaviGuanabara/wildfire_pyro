import os
import pandas as pd

# Arquivos de entrada
file_data = (
    "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire"
    "\\examples\\iowa_soil\\data\\daily\\processed\\tidy_isusm.csv"
)

file_meta = (
    "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire"
    "\\examples\\iowa_soil\\data\\daily\\raw\\stations.xlsx"
)

print("CSV existe?", os.path.exists(file_data))
print("XLSX existe?", os.path.exists(file_meta))

# ------------------------
# Load
# ------------------------
df_data = pd.read_csv(file_data)
df_meta = pd.read_excel(file_meta)

# ------------------------
# Normalize keys
# ------------------------
df_data["station"] = (
    df_data["station"].astype(str).str.strip().str.upper()
)
df_meta["ID"] = (
    df_meta["ID"].astype(str).str.strip().str.upper()
)

# ------------------------
# Ensure datetime (NO ordinal here)
# ------------------------
df_data["valid"] = pd.to_datetime(df_data["valid"], errors="raise")

# ------------------------
# Sanity check IDs
# ------------------------
ids_data = set(df_data["station"])
ids_meta = set(df_meta["ID"])

print("IDs só no CSV:", ids_data - ids_meta)
print("IDs só no Excel:", ids_meta - ids_data)

# ------------------------
# Merge
# ------------------------
df_merged = pd.merge(
    df_data,
    df_meta,
    left_on="station",
    right_on="ID",
    how="inner",
)

# ------------------------
# Save (datetime preserved)
# ------------------------
output_path = "estacoes_completas.csv"
df_merged.to_csv(output_path, index=False)

print(
    f"Arquivo final salvo com {len(df_merged)} linhas e {len(df_merged.columns)} colunas"
)
