import pandas as pd


if __name__ == "__main__":
    # Read the CSV file into a DataFrame
    df = pd.read_csv("BARCODE_Higher_filament_930am.csv")

    # Drop "channel" and "flag" columns
    df.drop(columns=["Channel", "Flags"], inplace=True, errors="ignore")
    
    df.isna().sum()  # Check for any NaN values
