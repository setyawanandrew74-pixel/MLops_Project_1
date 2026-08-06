import argparse
from pathlib import Path

import joblib
import pandas as pd

import config

# PENTING: import ini WAJIB ada meskipun terlihat "tidak dipakai".
# joblib.load() perlu tahu definisi ClusterSimilarity supaya bisa
# membaca ulang model yang sudah di-pickle. Kalau baris ini dihapus,
# load_model() akan error:
# "Can't get attribute 'ClusterSimilarity'".

from utils import ClusterSimilarity  # noqa: F401


# ======================
# LOAD MODEL
# ======================
def load_model(model_path: str | None = None):
    """
    Load model hasil training.

    Parameters
    ----------
    model_path : str | None, default=None
        Path model yang ingin digunakan.
        Jika None, akan menggunakan path dari config.py.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Model yang sudah dilatih.
    """
    path = Path(model_path) if model_path else config.MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan:\n{path}\n\n"
            "Silakan jalankan train.py terlebih dahulu."
        )

    return joblib.load(path)


# ======================
# PREDIKSI
# ======================
def predict_from_dataframe(model, df: pd.DataFrame) -> pd.Series:
    """
    Melakukan prediksi dari sebuah DataFrame.

    Parameters
    ----------
    model :
        Model hasil training.
    df : pd.DataFrame
        DataFrame yang memiliki kolom yang sama seperti data training.

    Returns
    -------
    pd.Series
        Hasil prediksi.
    """
    # Jika kolom target ikut terbawa, hapus terlebih dahulu
    if config.TARGET_COLUMN in df.columns:
        df = df.drop(columns=[config.TARGET_COLUMN])

    predictions = model.predict(df)

    return pd.Series(
        predictions,
        index=df.index,
        name="predicted_price",
    )


def predict_from_dict(model, data: dict) -> float:
    """
    Melakukan prediksi untuk satu rumah.

    Parameters
    ----------
    model :
        Model hasil training.
    data : dict
        Dictionary yang berisi fitur rumah.

    Returns
    -------
    float
        Harga rumah hasil prediksi.
    """
    df = pd.DataFrame([data])
    prediction = predict_from_dataframe(model, df)

    return float(prediction.iloc[0])


# ======================
# MAIN
# ======================
def main():
    parser = argparse.ArgumentParser(description="Prediksi harga rumah dari file CSV.")

    parser.add_argument(
        "--input",
        required=True,
        help="Path ke file CSV yang akan diprediksi.",
    )

    parser.add_argument(
        "--output",
        default=config.PREDICTION_PATH,
        help="Path untuk menyimpan hasil prediksi.",
    )

    args = parser.parse_args()

    model = load_model()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"File input tidak ditemukan:\n{input_path}")

    new_data = pd.read_csv(input_path)

    predictions = predict_from_dataframe(model, new_data)

    result_df = new_data.copy()
    result_df["predicted_price"] = predictions

    output_path = Path(args.output)
    result_df.to_csv(output_path, index=False)

    print(f"\nPrediksi selesai.")
    print(f"Hasil disimpan di: {output_path}")

    print("\nContoh hasil prediksi:")
    print(result_df.head(10))


if __name__ == "__main__":
    main()
