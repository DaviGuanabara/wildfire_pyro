import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib  # Para salvar o scaler
from clean_data import load_clean_dataset  # type: ignore


def load_clean_excel(file_path: str) -> pd.DataFrame:
    """
    Load the cleaned dataset from an Excel file.
    """
    df = pd.read_excel(file_path)
    print(f"✅ Loaded dataset with shape: {df.shape}")
    return df


def normalize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize numeric columns using Z-score, excluding spatial and ID columns.

    Returns:
        Tuple containing normalized dataframe and fitted scaler.
    """
    exclude_columns = ["Latitude1", "Longitude1",
                       "ID", "data", "Elevation [m]"]

    df_copy = df.copy()
    columns_to_normalize = [
        col for col in df.columns if col not in exclude_columns]

    scaler = StandardScaler()
    df_copy[columns_to_normalize] = scaler.fit_transform(
        df_copy[columns_to_normalize])

    print(f"✅ Normalized columns: {columns_to_normalize}")
    return df_copy, scaler


def save_scaler(scaler: StandardScaler, file_path: str) -> None:
    """
    Save the fitted scaler object to a file.

    Args:
        scaler (StandardScaler): The fitted scaler.
        file_path (str): Output file path for the scaler.
    """
    joblib.dump(scaler, file_path)
    print(f"✅ Scaler saved at: {file_path}")


def convert_date_to_int(df: pd.DataFrame, date_column: str = "data") -> pd.DataFrame:
    """
    Convert date column to integer ordinal representation.
    """
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    df_copy[date_column] = df_copy[date_column].apply(lambda x: x.toordinal())
    print(f"✅ Converted '{date_column}' to ordinal integers.")
    return df_copy


def save_preprocessed_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the preprocessed dataframe to an Excel file.
    """
    df.to_excel(file_path, index=False)
    print(f"✅ Preprocessed data saved at: {file_path}")


if __name__ == "__main__":
    # Load clean data
    df_final = load_clean_dataset(force_clean=False)

    # Convert date to int
    df_final = convert_date_to_int(df_final, date_column="data")

    # Normalize and get scaler
    df_final, scaler = normalize_dataframe(df_final)

    # Save preprocessed data
    save_preprocessed_data(
        df_final, "./data/ISU_Soil_Moisture_Network/dataset_preprocessed.xlsx")

    # Save scaler
    save_scaler(scaler, "./data/ISU_Soil_Moisture_Network/scaler.pkl")
