"""
transformers.py
Berisi "resep" preprocessing: custom transformer class dan pipeline builder.
File ini dipakai BERSAMA oleh train.py dan predict.py, supaya cara
menyiapkan data saat training dan saat prediksi selalu konsisten.

PENTING: ClusterSimilarity harus tetap ada di sini (bukan di train.py atau
predict.py) karena model yang di-pickle (joblib) menyimpan referensi ke
lokasi class ini. Kalau class-nya pindah/hilang, model gagal di-load.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder

import config


# ======================
# CUSTOM TRANSFORMER
# ======================
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self in fit method

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"cluster_similarity_{i}" for i in range(self.n_clusters)]


# ======================
# FUNGSI & SUB-PIPELINE
# ======================
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_names(function_transformer, feature_names_in):
    return ["ratio"]


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_names),
        StandardScaler(),
    )


def log_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler(),
    )


def cat_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )


# ======================
# GABUNGAN JADI SATU PREPROCESSING PIPELINE
# ======================
def preprocessing_pipeline():
    """Menyusun ColumnTransformer lengkap berdasarkan kolom-kolom di config.py.
    Dipakai baik saat training maupun saat prediksi, supaya transformasi
    datanya selalu sama persis."""
    default_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

    cluster_simil = ClusterSimilarity(
        n_clusters=config.CLUSTER_N_CLUSTERS_DEFAULT,
        gamma=config.CLUSTER_GAMMA,
        random_state=config.RANDOM_STATE,
    )

    preprocessing = ColumnTransformer(
        [
            ("bedrooms_per_room", ratio_pipeline(), config.BEDROOMS_PER_ROOM_COLS),
            ("rooms_per_household", ratio_pipeline(), config.ROOMS_PER_HOUSEHOLD_COLS),
            (
                "people_per_household",
                ratio_pipeline(),
                config.PEOPLE_PER_HOUSEHOLD_COLS,
            ),
            ("log", log_pipeline(), config.LOG_COLS),
            ("geo", cluster_simil, config.GEO_COLS),
            ("cat", cat_pipeline(), make_column_selector(dtype_include=object)),
        ],
        remainder=default_pipeline,
    )
    return preprocessing
