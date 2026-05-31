"""
automate_Nama-siswa.py
======================
Script otomatisasi preprocessing dataset Sports Car.
Mengembalikan dataset yang sudah siap dilatih (train-ready).

Cara pakai:
    python automate_Nama-siswa.py
    python automate_Nama-siswa.py --input ../dataset_raw.csv --output dataset_preprocessing/dataset_preprocessing.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Muat dataset dari file CSV."""
    df = pd.read_csv(filepath)
    print(f"[LOAD] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


# ─────────────────────────────────────────────
# 2. KONVERSI TIPE DATA
# ─────────────────────────────────────────────
def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Konversi kolom numerik yang masih bertipe string."""
    df = df.copy()

    numeric_cols = [
        "Engine Size (L)",
        "Horsepower",
        "Torque (lb-ft)",
        "0-60 MPH Time (seconds)",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Price: hapus koma ribuan lalu konversi ke float
    df["Price (in USD)"] = (
        df["Price (in USD)"].str.replace(",", "").astype(float)
    )

    print("[DTYPE] Konversi tipe data selesai.")
    return df


# ─────────────────────────────────────────────
# 3. HAPUS DUPLIKAT
# ─────────────────────────────────────────────
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus baris duplikat."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[DEDUP] Duplikat dihapus: {before - after} baris. Tersisa: {after}")
    return df


# ─────────────────────────────────────────────
# 4. HANDLE MISSING VALUES
# ─────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Isi missing values numerik dengan nilai median kolom."""
    df = df.copy()
    numeric_cols = [
        "Engine Size (L)",
        "Horsepower",
        "Torque (lb-ft)",
        "0-60 MPH Time (seconds)",
    ]
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[MISSING] '{col}': {missing_count} nilai diisi dengan median ({median_val:.2f})")
    print("[MISSING] Penanganan missing values selesai.")
    return df


# ─────────────────────────────────────────────
# 5. HAPUS OUTLIER (IQR)
# ─────────────────────────────────────────────
def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Hapus outlier menggunakan metode IQR (1.5 * IQR)."""
    df = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        after = len(df)
        print(f"[OUTLIER] '{col}': {before - after} baris dihapus (lower={lower:.2f}, upper={upper:.2f})")
    print(f"[OUTLIER] Baris tersisa setelah outlier removal: {len(df)}")
    return df


# ─────────────────────────────────────────────
# 6. ENCODING FITUR KATEGORIKAL
# ─────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode kolom kategorikal Car Make dan Car Model dengan LabelEncoder."""
    df = df.copy()
    le_make = LabelEncoder()
    le_model = LabelEncoder()

    df["Car Make Encoded"] = le_make.fit_transform(df["Car Make"])
    df["Car Model Encoded"] = le_model.fit_transform(df["Car Model"])

    df = df.drop(columns=["Car Make", "Car Model"])
    print("[ENCODE] Label encoding pada 'Car Make' dan 'Car Model' selesai.")
    return df


# ─────────────────────────────────────────────
# 7. FEATURE SCALING
# ─────────────────────────────────────────────
def scale_features(df: pd.DataFrame, target_col: str = "Price (in USD)") -> pd.DataFrame:
    """Scaling fitur dengan StandardScaler (kecuali target)."""
    df = df.copy()
    feature_cols = [c for c in df.columns if c != target_col]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print(f"[SCALE] StandardScaler diterapkan pada {len(feature_cols)} fitur.")
    return df


# ─────────────────────────────────────────────
# 8. SIMPAN HASIL
# ─────────────────────────────────────────────
def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Simpan dataset yang sudah diproses ke file CSV."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SAVE] Dataset preprocessing tersimpan di: {output_path}")
    print(f"[SAVE] Shape final: {df.shape}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def preprocess_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Pipeline lengkap preprocessing data.

    Parameters
    ----------
    input_path  : str  – path ke file dataset raw (.csv)
    output_path : str  – path output dataset preprocessing (.csv)

    Returns
    -------
    pd.DataFrame – dataset siap latih
    """
    print("=" * 50)
    print("  PIPELINE PREPROCESSING SPORTS CAR DATASET")
    print("=" * 50)

    # Step 1: Load
    df = load_data(input_path)

    # Step 2: Konversi dtype
    df = convert_dtypes(df)

    # Step 3: Hapus duplikat
    df = drop_duplicates(df)

    # Step 4: Handle missing values
    df = handle_missing(df)

    # Step 5: Hapus outlier
    outlier_cols = ["Horsepower", "Torque (lb-ft)", "Engine Size (L)", "Price (in USD)"]
    df = remove_outliers_iqr(df, outlier_cols)

    # Step 6: Encoding
    df = encode_categoricals(df)

    # Step 7: Scaling
    df = scale_features(df, target_col="Price (in USD)")

    # Step 8: Simpan
    save_data(df, output_path)

    print("=" * 50)
    print("  PREPROCESSING SELESAI!")
    print("=" * 50)
    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate Preprocessing Sports Car Dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="../dataset_raw.csv",
        help="Path ke dataset raw (default: ../dataset_raw.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset_preprocessing/dataset_preprocessing.csv",
        help="Path output dataset preprocessing",
    )
    args = parser.parse_args()

    preprocess_pipeline(input_path=args.input, output_path=args.output)
