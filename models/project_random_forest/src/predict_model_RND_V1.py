"""
predict.py
Script untuk melakukan prediksi harga rumah menggunakan model yang
sudah disimpan oleh train.py.

Cara pakai lewat terminal:
    python predict.py --input data_baru.csv --output hasil_prediksi.csv

Atau dipakai sebagai modul di script/aplikasi lain:
    from predict import load_model, predict_from_dataframe
    model = load_model()
    hasil = predict_from_dataframe(model, df_baru)
"""

import argparse
import pandas as pd
import joblib

import config

# PENTING: import ini WAJIB ada meskipun terlihat "tidak dipakai".
# joblib.load() perlu tahu definisi ClusterSimilarity supaya bisa
# membaca ulang model yang sudah di-pickle. Kalau baris ini dihapus,
# load_model() akan error: "Can't get attribute 'ClusterSimilarity'".

from utils import ClusterSimilarity  # noqa: F401


# ======================
# LOAD MODEL
# ======================
def load_model(model_path: str = None):
    """Load model hasil training. Default ambil path dari config.py."""
    path = model_path or config.MODEL_OUTPUT_PATH
    model = joblib.load(path)
    return model


# ======================
# PREDIKSI
# ======================
def predict_from_dataframe(model, df: pd.DataFrame) -> pd.Series:
    """
    Prediksi harga rumah dari DataFrame.
    df harus punya kolom-kolom yang sama seperti data training
    (kecuali kolom target 'median_house_value', yang tidak diperlukan).
    """
    # kalau target kolom kebawa ikut (misal user lupa drop), buang dulu
    if config.TARGET_COL in df.columns:
        df = df.drop(columns=[config.TARGET_COL])

    predictions = model.predict(df)
    return pd.Series(predictions, index=df.index, name="predicted_price")


def predict_from_dict(model, data: dict) -> float:
    """
    Prediksi untuk SATU rumah, input berupa dictionary.
    Contoh:
        data = {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY",
        }
    """
    df = pd.DataFrame([data])
    result = predict_from_dataframe(model, df)
    return float(result.iloc[0])


# ======================
# CLI (dipanggil lewat terminal)
# ======================
def main():
    parser = argparse.ArgumentParser(description="Prediksi harga rumah dari file CSV.")
    parser.add_argument("--input", required=True, help="Path ke CSV data baru")
    parser.add_argument(
        "--output", default="predictions.csv", help="Path CSV hasil prediksi"
    )
    args = parser.parse_args()

    model = load_model()

    new_data = pd.read_csv(args.input)
    predictions = predict_from_dataframe(model, new_data)

    result_df = new_data.copy()
    result_df["predicted_price"] = predictions
    result_df.to_csv(args.output, index=False)

    print(f"Prediksi selesai. Hasil disimpan di: {args.output}")
    print(result_df[["predicted_price"]].head(10))


if __name__ == "__main__":
    main()
