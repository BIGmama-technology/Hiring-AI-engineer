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


def load_data_set(file_path : str, feature_column : str, target_column : str, normalized : bool=True, convert_to_int : bool=False) -> tuple[Union[np.ndarray, pd.Series], np.ndarray, Union[np.ndarray, pd.Series]]:
    """
        Load any dataset from a CSV file format

        Args:
            file_path (str): The path to the CSV file
            feature_column(str): The name of the feature column
            target_column(str): The name of the target column
            normalized (bool): Whether to normalize the data or not
            convert_to_int (bool): Whether to convert the feature to int64

        Returns:
            Tuple: a tuple containing X, y, and optionally X_normalized
    """
    # Load data into a DataFrame
    df = pd.read_csv(file_path)

    # Handle different data types for the feature column
    if df[feature_column].dtype == 'object':
        # convert to datetime
        df[feature_column] = pd.to_datetime(df[feature_column], errors='coerce')
    if convert_to_int:
        # convert the feature to int64
        df[feature_column] = df[feature_column].astype('int64')

    # Prepare the data
    X = df[feature_column].values.reshape(-1, 1)
    y = df[target_column].values

    # Normalize the data for numerical stability if specified
    if normalized:
        X_normalized = normalize(X)
        return X, y, X_normalized
    else:
        return X, y