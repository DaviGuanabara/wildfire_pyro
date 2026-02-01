import pandas as pd
import numpy as np

# =============================
# CONFIG
# =============================
DATA_PATH = "train.csv"
TIME_COL = "valid"
ID_COL = "station"
SOLAR_COL = "in_solar_mj"

MAX_DELTA_TIME = 10  # mesmo conceito do Adapter
MAX_SAMPLES = None  # None = varre tudo | ou ex: 5000 para testar rápido

# =============================
# LOAD
# =============================
print("🔍 Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print()

# =============================
# BASIC COLUMN CHECK
# =============================
print("🔍 Checking column existence...")
if SOLAR_COL not in df.columns:
    raise RuntimeError(f"❌ Column {SOLAR_COL} DOES NOT EXIST in CSV.")

print("✅ Column exists.")
print()

# =============================
# TYPE + NaN CHECK
# =============================
print("🔍 Checking dtype and NaN...")
print("dtype:", df[SOLAR_COL].dtype)
print("NaN count:", df[SOLAR_COL].isna().sum())

if df[SOLAR_COL].isna().any():
    print("⚠️ Rows with NaN in in_solar_mj:")
    print(df[df[SOLAR_COL].isna()].head())
    raise RuntimeError("❌ NaN detected in in_solar_mj.")

print("✅ No NaN detected.")
print()

# =============================
# TIME NORMALIZATION (as in your code)
# =============================
df[TIME_COL] = pd.to_datetime(df[TIME_COL]).map(pd.Timestamp.toordinal)

# =============================
# MAIN AUDIT LOOP
# =============================
print("🔍 Auditing neighbor subsets...\n")

times = np.sort(df[TIME_COL].unique())
checked = 0

for idx, row in df.iterrows():

    if MAX_SAMPLES is not None and checked >= MAX_SAMPLES:
        break

    t = row[TIME_COL]
    station = row[ID_COL]

    # Candidates: before index (causal), different station, time window
    candidates = df.loc[:idx - 1] if idx > 0 else df.iloc[[]] # type: ignore

    candidates = candidates[candidates[ID_COL] != station]
    candidates = candidates[np.abs(candidates[TIME_COL] - t) <= MAX_DELTA_TIME]

    if candidates.empty:
        checked += 1
        continue

    # 🔥 CRITICAL CHECK
    if SOLAR_COL not in candidates.columns:
        print("❌ COLUMN MISSING IN NEIGHBORS")
        print("Row index:", idx)
        print("Time:", t)
        print("Candidates columns:", candidates.columns.tolist())
        raise RuntimeError("Fatal: column disappeared.")

    if candidates[SOLAR_COL].isna().any():
        print("❌ NaN FOUND IN NEIGHBORS")
        print("Row index:", idx)
        print("Time:", t)
        print(candidates[[SOLAR_COL]].head())
        raise RuntimeError("Fatal: NaN appeared in neighbors.")

    checked += 1

print()
print("✅ AUDIT COMPLETE")
print(f"Checked {checked} samples.")
print("✅ in_solar_mj is ALWAYS present and valid in all neighbor subsets.")
