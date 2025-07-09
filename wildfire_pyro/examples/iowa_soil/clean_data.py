import pandas as pd
import os

def merge_station_metadata(df_main: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Merge main weather dataframe with station metadata using 'ID' as the key.

    Note:
        In df_main, the station ID is named 'station', and in df_meta it is 'ID'.
        After merge, the column should be named 'ID'.

    Returns:
        Merged dataframe with additional location columns.
    """
    df_main = df_main.rename(columns={'station': 'ID'})
    df_merged = pd.merge(df_main, df_meta, on='ID', how='left')
    return df_merged


def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are completely empty (all NaN)."""
    return df.dropna(axis=1, how="all")


def remove_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove all columns that end with '_f' (quality flags)."""
    cols_to_remove = [col for col in df.columns if col.endswith('_f')]
    return df.drop(columns=cols_to_remove)


def remove_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove specific columns listed as unnecessary metadata."""
    columns_to_remove = ["Archive Ends", "IEM Network", "Attributes", "Archive Begins", "Station Name"]
    return df.drop(columns=columns_to_remove, errors="ignore")


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns using a predefined dictionary.

    Example: 'valid' becomes 'data'.
    """
    rename_dict = {
        "valid": "data"
    }
    return df.rename(columns=rename_dict)


def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows containing any missing (NaN) values."""
    return df.dropna()


def save_to_excel(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to an Excel file."""
    df.to_excel(file_path, index=False)
    print(f"✅ File successfully saved at: {file_path}")


def load_clean_dataset(force_clean: bool = False) -> pd.DataFrame:
    """
    Load the cleaned dataset. If a cleaned file exists and force_clean is False,
    load it directly. Otherwise, process from raw files.

    Steps:
        #UNITE
        #RENAME
        #REMOVE EMPTY COLUMNS
        #REMOVE FLAG COLUMNS
        #REMOVE ROWS THAT MISS VALUES
        #SAVE DATAFRAME

    Args:
        force_clean (bool): Whether to force re-processing from raw files.

    Returns:
        pd.DataFrame: The final cleaned dataframe.
    """
    cleaned_file_path = "./data/ISU_Soil_Moisture_Network/dataset_cleaned.xlsx"

    if os.path.exists(cleaned_file_path) and not force_clean:
        print("✅ Cleaned dataset found. Loading...")
        return pd.read_excel(cleaned_file_path)
    
    print("⚙️ Processing raw data...")

    # Load raw data
    df_weather = pd.read_excel("./data/ISU_Soil_Moisture_Network/isusm.xlsx")
    df_stations = pd.read_excel("./data/ISU_Soil_Moisture_Network/stations.xlsx")

    # UNITE
    df = merge_station_metadata(df_weather, df_stations)

    # RENAME
    df = rename_columns(df)

    # REMOVE EMPTY COLUMNS
    df = remove_empty_columns(df)

    # REMOVE FLAG COLUMNS
    df = remove_flag_columns(df)

    # REMOVE COLUMNS (unwanted metadata columns)
    df = remove_columns(df)

    # REMOVE ROWS THAT MISS VALUES
    df = remove_missing_values(df)

    # SAVE DATAFRAME
    save_to_excel(df, cleaned_file_path)

    return df


def show_clean_dataset_summary(df: pd.DataFrame) -> None:
    """
    Show summary indexes and metadata for the cleaned dataset.

    Prints:
        - List of unique station IDs.
        - List of station names (if available).
        - Number of records per station.
        - Date range (if 'data' column exists).
        - Total number of rows.
    """
    print("🗺️ Dataset Summary")
    print("-------------------------------")
    
    # Station IDs
    if 'ID' in df.columns:
        station_ids = df['ID'].unique()
        print(f"✅ Stations IDs present ({len(station_ids)}): {station_ids.tolist()}")
        
        # Count per station
        print("\n📊 Number of records per station:")
        print(df['ID'].value_counts())
    
    # Station names
    if 'Station Name' in df.columns:
        station_names = df['Station Name'].unique()
        print(f"\n🏷️ Station names ({len(station_names)}): {station_names.tolist()}")
    
    # Date range
    if 'data' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['data']):
            df['data'] = pd.to_datetime(df['data'])
        min_date = df['data'].min()
        max_date = df['data'].max()
        print(f"\n⏳ Date range: {min_date.date()} to {max_date.date()}")
    
    # Total rows
    print(f"\n🔢 Total rows: {len(df)}")
    print("-------------------------------")


if __name__ == "__main__":
    df_final = load_clean_dataset(force_clean=False)
    show_clean_dataset_summary(df_final)
