"""
config.py
Semua konfigurasi (path, parameter, hyperparameter) untuk project
housing price prediction. Tujuannya: train.py cukup import dari sini,
jadi kalau mau ganti setting, cukup edit file ini saja.
"""

# ======================
# 1. PATH
# ======================
DATA_PATH = r"D:\USER\git_github\MLOps_project_1\data\raw\housing.csv"
MODEL_OUTPUT_PATH = (
    r"D:\USER\git_github\MLOps_project_1\models\project_random_forest"
    r"\model_V1\my_california_housing_model_rnd.pkl"
)
# Test set disimpan sebagai CSV supaya bisa dipakai ulang oleh predict.py
# maupun evaluate.py, tanpa perlu split ulang dari housing.csv setiap kali.
TEST_SET_OUTPUT_PATH = r"D:\USER\git_github\MLOps_project_1\data\processed\test_set.csv"

# ======================
# 2. TRAIN-TEST SPLIT
# ======================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Untuk stratified split berdasarkan median_income
INCOME_CAT_BINS = [0.0, 1.5, 3.0, 4.5, 6.0, float("inf")]
INCOME_CAT_LABELS = [1, 2, 3, 4, 5]

# ======================
# 3. KOLOM-KOLOM FITUR
# ======================
# Kolom untuk masing-masing ratio pipeline
BEDROOMS_PER_ROOM_COLS = ["total_bedrooms", "total_rooms"]
ROOMS_PER_HOUSEHOLD_COLS = ["total_rooms", "households"]
PEOPLE_PER_HOUSEHOLD_COLS = ["population", "households"]

# Kolom untuk log pipeline
LOG_COLS = [
    "total_bedrooms",
    "total_rooms",
    "households",
    "population",
    "median_income",
]

# Kolom untuk geo clustering (ClusterSimilarity)
GEO_COLS = ["latitude", "longitude"]

# Target column
TARGET_COL = "median_house_value"

# ======================
# 4. HYPERPARAMETER CLUSTERSIMILARITY (default sebelum tuning)
# ======================
CLUSTER_N_CLUSTERS_DEFAULT = 10
CLUSTER_GAMMA = 1.0

# ======================
# 5. HYPERPARAMETER RANDOM FOREST (default)
# ======================
RF_RANDOM_STATE = 42

# ======================
# 6. RANDOMIZED SEARCH CV
# ======================
from scipy.stats import randint

PARAM_DISTRIBS = {
    "preprocessing__geo__n_clusters": randint(low=3, high=50),
    "random_forest__max_features": randint(low=2, high=25),
}

CV_N_ITER = 10
CV_FOLDS = 3
CV_SCORING = "neg_root_mean_squared_error"
