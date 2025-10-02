import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Carregar dataset já merged
df = pd.read_csv(
    "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\wildfire_pyro\\examples\\iowa_soil\\data\\processed\\tidy_isusm_stations.csv"
)

# Converter 'valid' para datetime
if pd.api.types.is_integer_dtype(df["valid"]):
    df["valid"] = pd.to_datetime(df["valid"].apply(
        lambda x: pd.Timestamp.fromordinal(int(x))))
else:
    df["valid"] = pd.to_datetime(df["valid"])

# Calcular diferença normalizada (fração do dia)
df["diff"] = np.abs(df["out_lwmwet_1_tot"] - df["out_lwmwet_2_tot"]) / 1440.0


# Agrupar por estação e calcular estatísticas
summary = (
    df.groupby("station")["diff"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "mean_diff", "std": "std_diff", "count": "num_registros"})
)

summary_corr = (
    df.groupby("station")[["out_lwmwet_1_tot", "out_lwmwet_2_tot"]]
    .corr()
    .iloc[0::2, -1]   # pega só correlação linha-coluna certa
    .reset_index()
    .rename(columns={"out_lwmwet_2_tot": "corr_wet1_wet2"})
    [["station", "corr_wet1_wet2"]]
)

# juntar na summary
summary = summary.merge(summary_corr, on="station", how="left")

# Ordenar por maior média de diferença
summary = summary.sort_values("mean_diff", ascending=False)

# Salvar para CSV
summary.to_csv("estatisticas_por_estacao.csv", index=False)

print(summary.head(10))  # mostra as 10 estações com maior diferença


# 🔹 Loop por estação
stations = df["station"].unique()

for st in stations:
    df_station = df[df["station"] == st].copy()

    # Filtrar intervalo (exemplo: 2023–2023)
    df_period = df_station[(df_station["valid"] >= "2023-01-01")
                           & (df_station["valid"] <= "2023-12-31")]

    if df_period.empty:
        continue

    # Plotar diferença
    plt.figure(figsize=(14, 6))

    plt.plot(df_period["valid"], df_period["diff"], color="purple",
             linewidth=1.5, label=f"{st} – Diferença Normalizada")

    plt.fill_between(df_period["valid"],
                     df_period["diff"], color="purple", alpha=0.3)

    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Data")
    plt.ylabel("Diferença de molhamento (fração do dia)")
    plt.title(
        f"Estação {st}: Diferença entre sensores de molhamento foliar (Wet1 - Wet2)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
