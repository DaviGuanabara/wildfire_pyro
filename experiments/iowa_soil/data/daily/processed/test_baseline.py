import pandas as pd
df = pd.read_csv("dataset_with_baseline.csv")

diff = df["baseline"] - df["out_lwmwet_1_tot"]
print(diff.abs().describe())
