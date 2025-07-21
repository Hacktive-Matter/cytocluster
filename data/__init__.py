import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_data():
    df = pd.read_csv("data/BARCODE_Higher_filament_930am.csv")

    # Drop "channel" and "flag" columns
    df.drop(columns=["Filename", "Channel", "Flags"], inplace=True, errors="ignore")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[df.columns])

    return df, X_scaled
