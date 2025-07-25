from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re

exclude_columns = [
    "File",
    "Channel",
    "Flags",
]


def get_c1_data() -> tuple[pd.DataFrame, np.ndarray]:
    df_cyan_c1 = pd.read_csv("dataset_4/Cyan C1 Summary.csv")
    df_yellow_c1 = pd.read_csv("dataset_4/Yellow C1 Summary.csv")

    df_c1 = pd.concat([df_cyan_c1, df_yellow_c1], ignore_index=True)
    df_c1.drop(columns=exclude_columns, inplace=True, errors="ignore")

    # Scale only the feature data
    scaler = StandardScaler()
    X_scaled_c1 = scaler.fit_transform(df_c1)
    df_c1 = pd.DataFrame(X_scaled_c1, columns=df_c1.columns)

    return df_c1, X_scaled_c1


def get_c2_data() -> tuple[pd.DataFrame, np.ndarray]:
    df_cyan_c2 = pd.read_csv("dataset_4/Cyan C2 Summary.csv")
    df_yellow_c2 = pd.read_csv("dataset_4/Yellow C2 Summary.csv")

    df_c2 = pd.concat([df_cyan_c2, df_yellow_c2], ignore_index=True)
    df_c2.drop(columns=exclude_columns, inplace=True, errors="ignore")

    scaler = StandardScaler()
    X_scaled_c2 = scaler.fit_transform(df_c2)
    df_c2 = pd.DataFrame(X_scaled_c2, columns=df_c2.columns)

    return df_c2, X_scaled_c2


def get_all_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Get all data from C1 and C2 datasets.
    
    Returns:
        df_c1: DataFrame for C1 data
        X_c1: Scaled feature matrix for C1
        df_c2: DataFrame for C2 data
        X_c2: Scaled feature matrix for C2
    """
    df_c1, X_c1 = get_c1_data()
    df_c2, X_c2 = get_c2_data()
    
    df = pd.concat([df_c1, df_c2], ignore_index=True)
    X = np.vstack([X_c1, X_c2])

    return df, X