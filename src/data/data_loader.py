import numpy as np
import pandas as pd
from typing import Union

def normalize(x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Normalize a pandas series or numpy array
    Args:
        x: The data to normalize

    Returns:
        Normalized data
    """
    return ((x - x.mean()) / x.std() )


def load_mauna_loa_atmospheric_co2(file_path):
    # Load data into a DataFrame
    df = pd.read_csv(file_path)
    # Prepare the data
    X = df[["decimal date"]].values.reshape(-1, 1)
    y = df["average"].values

    # Normalize the data for numerical stability
    X_normalized = normalize(X)
    return X, y, X_normalized


def load_international_airline_passengers(file_path):
    # Load  data into a DataFrame
    df_airpassengers = pd.read_csv(file_path)

    # Prepare the data
    X_airpassengers = (
        pd.to_datetime(df_airpassengers["Month"])
        .dt.to_period("M")
        .astype("int64")
        .values.reshape(-1, 1)
    )
    y_airpassengers = df_airpassengers["Passengers"].values

    # Normalize the data for numerical stability
    X_airpassengers_normalized = normalize(X_airpassengers)

    return X_airpassengers, y_airpassengers, X_airpassengers_normalized


def load_data_set(file_path, feature_column, target_column, normalized = True):
    """
    Load any dataset from a CSV file format

    Args:
        file_path (str): The path to the CSV file
        feature_column(str): The name of the feature column
        target_column(str): The name of the target column
        normalized (bool): Whether to normalize the data or not

    Returns:
        Tuple: a tuple containing X, y, and optionally X_normalized
    """
    # Load data into a DataFrame
    df = pd.read_csv(file_path)

    # Prepare the data
    #X = df[feature_column].values.reshape(-1, 1)
    X = pd.to_datetime(df[feature_column]).astype("int64").values.reshape(-1, 1)
    y = df[target_column].values
    # Normalize the data for numerical stability
    if normalized:
        X_normalized = normalize(X)
        return X, y, X_normalized
    else:
        return X, y


