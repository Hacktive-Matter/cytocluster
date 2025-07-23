import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

exclude_columns = ["Filename", "Channel", "Flags"]


# def get_data(channel=None) -> tuple[pd.DataFrame, np.ndarray]:
#     # df = pd.read_csv("dataset_1/Dataset 1.csv")
#     df = pd.read_csv("dataset_1/PNAS Nexus 2023 Summary -- For Ayan.csv")

#     # Split df where Filename contains C=1
#     df_c1 = df[df["Channel"] == 0].reset_index(drop=True)
#     df_c2 = df[df["Channel"] == 1].reset_index(drop=True)
#     # df_c1 = df[df["Filename"].str.contains("C=0")]
#     # df_c2 = df[df["Filename"].str.contains("C=1")]

#     df_c1.drop(columns=exclude_columns, inplace=True, errors="ignore")
#     df_c2.drop(columns=exclude_columns, inplace=True, errors="ignore")

#     metric_columns = ["Crosslinkers", "Myosin", "Mode"]

#     if channel is None:
#         df = pd.concat([df_c1, df_c2], ignore_index=True)
#     if channel == 1:
#         df = df_c1
#     elif channel == 2:
#         df = df_c2

#     # Scale only the feature data
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df)
#     df = pd.DataFrame(X_scaled, columns=df.columns)

#     return df, X_scaled


def get_data(channel=None) -> tuple[pd.DataFrame, np.ndarray]:
    # df = pd.read_csv("dataset_1/Dataset 1.csv")
    df = pd.read_csv("dataset_1/PNAS Nexus 2023 Summary -- For Ayan.csv")

    # Split df where Channel == 0 or 1
    df_c1 = df[df["Channel"] == 0].reset_index(drop=True)
    df_c2 = df[df["Channel"] == 1].reset_index(drop=True)

    # Remove exclude_columns from both dataframes
    df_c1.drop(columns=exclude_columns, inplace=True, errors="ignore")
    df_c2.drop(columns=exclude_columns, inplace=True, errors="ignore")

    # Define metric columns that should be kept in df but not scaled
    metric_columns = ["Crosslinkers", "Myosin", "Mode"]

    # Select the appropriate channel data
    if channel is None:
        df = pd.concat([df_c1, df_c2], ignore_index=True)
    elif channel == 1:
        df = df_c1
    elif channel == 2:
        df = df_c2

    # Create feature columns for scaling (exclude metric columns)
    feature_columns = [col for col in df.columns if col not in metric_columns]
    X_features = df[feature_columns]

    # Scale only the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Create a new dataframe with scaled features and original metric columns
    df_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

    # Add back the metric columns (unscaled)
    for col in metric_columns:
        if col in df.columns:
            df_scaled[col] = df[col].values

    # Reorder columns to match original order (optional)
    df_scaled = df_scaled[df.columns]

    return df_scaled, X_scaled
