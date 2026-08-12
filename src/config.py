"""
config.py
Konfigurasi terpusat untuk project Housing Price Prediction.

Semua pengaturan project berada di sini:
- Path file & folder
- Pilihan algoritma
- Versi model
- Hyperparameter
- Parameter training

Dengan desain ini train.py dan predict.py tidak perlu diubah
ketika ingin mengganti algoritma ataupun versi model.
"""

from pathlib import Path

from scipy.stats import randint

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# ==========================================================
# 1. PILIH MODEL YANG AKTIF
# ==========================================================

ACTIVE_MODEL = "random_forest"  # linear_regression | decision_tree | random_forest
MODEL_VERSION = "model_V1"


# ==========================================================
# 2. BASE DIRECTORY
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"


# ==========================================================
# 3. FILE PATH
# ==========================================================

DATA_PATH = RAW_DATA_DIR / "housing.csv"

TEST_SET_PATH = PROCESSED_DIR / "test_set.csv"

MODEL_DIR = MODELS_DIR / f"project_{ACTIVE_MODEL}" / MODEL_VERSION

MODEL_PATH = MODEL_DIR / "model.pkl"

PREDICTION_PATH = MODEL_DIR / "predictions_test.csv"
# ==========================================================
# 4. TARGET
# ==========================================================

TARGET_COLUMN = "median_house_value"


# ==========================================================
# 5. TRAIN TEST SPLIT
# ==========================================================

TEST_SIZE = 0.2
RANDOM_STATE = 42

INCOME_CAT_BINS = [0.0, 1.5, 3.0, 4.5, 6.0, float("inf")]
INCOME_CAT_LABELS = [1, 2, 3, 4, 5]


# ==========================================================
# 6. PREPROCESSING
# ==========================================================

BEDROOMS_PER_ROOM_COLS = [
    "total_bedrooms",
    "total_rooms",
]

ROOMS_PER_HOUSEHOLD_COLS = [
    "total_rooms",
    "households",
]

PEOPLE_PER_HOUSEHOLD_COLS = [
    "population",
    "households",
]

LOG_COLS = [
    "total_bedrooms",
    "total_rooms",
    "households",
    "population",
    "median_income",
]

GEO_COLS = [
    "latitude",
    "longitude",
]


# ==========================================================
# 7. CLUSTER SIMILARITY
# ==========================================================

CLUSTER_N_CLUSTERS_DEFAULT = 10
CLUSTER_GAMMA = 1.0


# ==========================================================
# 8. MODEL CONFIGURATION
# ==========================================================

MODEL_CONFIG = {
    "linear_regression": {
        "model": LinearRegression(),
        "params": {
            "preprocessing__geo__n_clusters": randint(3, 50),
        },
    },
    "decision_tree": {
        "model": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "params": {
            "preprocessing__geo__n_clusters": randint(3, 50),
            "model__max_depth": randint(2, 30),
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 20),
        },
    },
    "random_forest": {
        "model": RandomForestRegressor(random_state=RANDOM_STATE),
        "params": {
            "preprocessing__geo__n_clusters": randint(3, 50),
            "model__max_features": randint(2, 25),
        },
    },
}


# ==========================================================
# 9. TRAINING CONFIG
# ==========================================================

CV_N_ITER = 10
CV_FOLDS = 3

CV_SCORING = "neg_root_mean_squared_error"
