import csv
import os
from pathlib import Path


def convert_nonstandard_csv(input_file, output_file):
    """
    Convert non-standard CSV format to proper CSV with filename column.

    Input format:
    - Line 1: Header row
    - Line 2: Filename
    - Line 3: Data row
    - Line 4: Empty line
    - Repeat lines 2-4 for each file
    """

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    # Remove empty lines for easier processing
    non_empty_lines = [line for line in lines if line]

    # First line is the header
    header = non_empty_lines[0].split(",")

    # Add filename column to header
    new_header = ["Filename"] + header

    # Prepare data rows
    data_rows = []

    # Process remaining lines in pairs (filename, data)
    i = 1
    while i < len(non_empty_lines):
        if i + 1 < len(non_empty_lines):
            filename = non_empty_lines[i]
            data_row = non_empty_lines[i + 1].split(",")

            # Extract just the filename from the full path
            filename_only = os.path.basename(filename)

            # Combine filename with data
            new_row = [filename_only] + data_row
            data_rows.append(new_row)

            i += 2  # Move to next filename
        else:
            break

    # Write to new CSV file
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(data_rows)

    print(f"Conversion complete!")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Processed {len(data_rows)} data rows")
    print(f"Columns: {len(new_header)}")


# Usage
if __name__ == "__main__":
    input_file = "data/BARCODE_Higher_filament_930am_original.csv"
    output_file = "data/BARCODE_Higher_filament_930am.csv"

    convert_nonstandard_csv(input_file, output_file)