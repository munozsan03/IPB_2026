import pandas as pd
from joblib import Parallel, delayed
from calculate_features import Substrate  # Make sure that calculate_features.py is in the same directory

# Hardcoded file paths
INPUT_CSV = "/raid/data/smunoz/catnip/data/subsrates_quick_test.csv"
OUTPUT_CSV = "/raid/data/smunoz/catnip/data/subsrates_with_features.csv"

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)

    def process_row(row):
        try:
            substrate = Substrate(row["SMILES"])
            substrate.get_geometry_and_basic_features()
            substrate.calculate_features()

            for feature_name, feature_value in substrate.features.items():
                row[feature_name] = feature_value

        except Exception as e:
            print(f"Error processing {row['SMILES']}, {e}")

        return row

    output_data = Parallel(n_jobs=-1)(delayed(process_row)(row) for _, row in df.iterrows())
    output_data = [row for row in output_data if row is not None]

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(OUTPUT_CSV, index=False)