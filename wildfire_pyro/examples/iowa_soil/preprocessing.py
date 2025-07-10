import pandas as pd
from sklearn.preprocessing import StandardScaler
from .clean_data import load_clean_dataset  # Ok se estiver no seu projeto local

def load_clean_excel(file_path: str) -> pd.DataFrame:
    """
    Load the cleaned dataset from an Excel file.

    Args:
        file_path (str): Path to the cleaned Excel file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    df = pd.read_excel(file_path)
    print(f"✅ Loaded dataset with shape: {df.shape}")
    return df


def normalize_dataframe(df: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
    """
    Normalize all columns using Z-score, excluding specified columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        exclude_columns (list): List of columns to exclude from normalization.

    Returns:
        pd.DataFrame: Normalized dataframe.
    """
    df_copy = df.copy()
    columns_to_normalize = [col for col in df.columns if col not in exclude_columns]

    scaler = StandardScaler()
    df_copy[columns_to_normalize] = scaler.fit_transform(df_copy[columns_to_normalize])

    print(f"✅ Normalized columns: {columns_to_normalize}")
    return df_copy


def convert_date_to_int(df: pd.DataFrame, date_column: str = "data") -> pd.DataFrame:
    """
    Convert date column to integer ordinal representation.

    Args:
        df (pd.DataFrame): Input dataframe.
        date_column (str): Name of the date column.

    Returns:
        pd.DataFrame: Dataframe with converted date column.
    """
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    df_copy[date_column] = df_copy[date_column].apply(lambda x: x.toordinal())
    print(f"✅ Converted '{date_column}' to ordinal integers.")
    return df_copy


def save_preprocessed_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the preprocessed dataframe to an Excel file.

    Args:
        df (pd.DataFrame): Preprocessed dataframe.
        file_path (str): Output Excel file path.
    """
    df.to_excel(file_path, index=False)
    print(f"✅ Preprocessed data saved at: {file_path}")


if __name__ == "__main__":
    # Load clean data
    df_final = load_clean_dataset(force_clean=False)

    # Convert date to int
    df_final = convert_date_to_int(df_final, date_column="data")

    # Define columns to exclude from normalization
    exclude_columns = ["Latitude1", "Longitude1", "ID", "data"]

    # Normalize
    df_final = normalize_dataframe(df_final, exclude_columns=exclude_columns)

    # Save preprocessed data
    save_preprocessed_data(df_final, "./data/ISU_Soil_Moisture_Network/dataset_preprocessed.xlsx")
