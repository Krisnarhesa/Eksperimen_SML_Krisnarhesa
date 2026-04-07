import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_data(path):
    logging.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logging.info(f"Data shape: {df.shape}")
    return df

def validate_data(df):
    logging.info("Validating dataset...")

    required_cols = ['Churn', 'tenure', 'TotalCharges']
    for col in required_cols:
        if col not in df.columns:
            logging.error(f"Kolom {col} tidak ditemukan!")
            raise ValueError(f"Kolom {col} tidak ditemukan di dataset!")

    logging.info("Validation passed")

def preprocess_data(df):
    logging.info("Starting preprocessing...")

    df = df.drop(columns=['customerID'], errors='ignore')

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    before = df.shape[0]
    df = df.dropna()
    logging.info(f"Drop NA: {before} -> {df.shape[0]}")

    before = df.shape[0]
    df = df.drop_duplicates()
    logging.info(f"Drop duplicates: {before} -> {df.shape[0]}")

    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, 100],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5+yr']
    )
    logging.info("Feature engineering: tenure_group created")

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    df = pd.get_dummies(df, drop_first=True)
    logging.info(f"After encoding shape: {df.shape}")

    target = df['Churn']
    df = df.drop('Churn', axis=1)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    df_final = pd.DataFrame(df_scaled, columns=df.columns)
    df_final['target'] = target.values

    logging.info(f"Final dataset shape: {df_final.shape}")

    return df_final

def save_data(df, path):
    df.to_csv(path, index=False)
    logging.info(f"Saved processed data to {path}")

def main():
    try:
        input_path = "dataset_raw/dataset.csv"
        output_path = "preprocessing/dataset_preprocessing.csv"

        df = load_data(input_path)
        validate_data(df)
        df_clean = preprocess_data(df)
        save_data(df_clean, output_path)

        logging.info("Preprocessing pipeline completed successfully!")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()