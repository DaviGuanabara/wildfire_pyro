import pandas as pd

# =============================
# CONFIG
# =============================
DATA_PATH = "dataset_with_baseline.csv"
MAX_EXAMPLES = 5

# =============================
# LOAD
# =============================
print("🔍 Loading dataset_with_baseline.csv...\n")
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print()

# =============================
# NaN AUDIT
# =============================
nan_summary = df.isna().sum()
nan_columns = nan_summary[nan_summary > 0]

if nan_columns.empty:
    print("✅ No NaN found in any column.")
    exit(0)

print("❌ NaN FOUND IN DATASET")
print("=" * 80)

for col, count in nan_columns.items():
    print(f"\n📌 Column: {col}")
    print(f"NaN count: {count}")

    examples = df[df[col].isna()].head(MAX_EXAMPLES)

    print("Examples:")
    print(
        examples[
            ["station", "valid", col]
            if "station" in df.columns and "valid" in df.columns
            else [col]
        ]
    )

print("\n" + "=" * 80)
print("⚠️ AUDIT FINISHED — DATASET CONTAINS MISSING VALUES")
