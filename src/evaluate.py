"""
evaluate.py
Evaluasi model akhir menggunakan test set yang sudah disisihkan sejak awal.
Menghitung RMSE dan bootstrap confidence interval untuk RMSE tersebut.
"""

import config
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import root_mean_squared_error

# Wajib diimpor meskipun kelihatan tidak dipakai langsung,
# supaya joblib bisa "membangun ulang" objek ClusterSimilarity saat load model.
from utils import ClusterSimilarity


def load_test_set(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load test set dan pisahkan fitur (X) dari target (y)."""
    test_set = pd.read_csv(config.TEST_SET_PATH)
    X_test = test_set.drop(columns=[config.TARGET_COLUMN])
    y_test = test_set[config.TARGET_COLUMN].copy()
    return X_test, y_test


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


def compute_rmse(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[float, np.ndarray]:
    """Hitung RMSE final, sekaligus balikin array residual (buat bootstrap)."""
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    squared_errors = (predictions - y_test.to_numpy()) ** 2
    return rmse, squared_errors


def bootstrap_rmse_ci(squared_errors: np.ndarray, confidence: float = 0.95):
    """Hitung confidence interval untuk RMSE pakai bootstrap resampling."""

    def rmse_stat(errors, axis=-1):
        return np.sqrt(np.mean(errors, axis=axis))

    result = stats.bootstrap(
        (squared_errors,),
        rmse_stat,
        confidence_level=confidence,
        method="Bca",
    )
    return result.confidence_interval


def main():
    print("Model aktif:", config.ACTIVE_MODEL)

    print("Memuat test set...")
    X_test, y_test = load_test_set(config.TEST_SET_PATH)

    print("Memuat model...")
    model = load_model(config.MODEL_PATH)

    print("Menghitung RMSE...")
    rmse, squared_errors = compute_rmse(model, X_test, y_test)
    print(f"RMSE pada test set: {rmse:,.2f}")

    print("Menghitung bootstrap confidence interval...")
    ci = bootstrap_rmse_ci(squared_errors)
    print(f"95% CI untuk RMSE: [{ci.low:,.2f}, {ci.high:,.2f}]")


if __name__ == "__main__":
    main()
