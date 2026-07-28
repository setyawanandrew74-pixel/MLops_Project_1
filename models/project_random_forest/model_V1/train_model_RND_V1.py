"""
train.py
Script utama untuk training model prediksi harga rumah.
Semua path, hyperparameter, dan daftar kolom diambil dari config.py
supaya file ini fokus ke alur logika saja.
"""

# step 1: import library yang diperlukan
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

import config


# ======================
# CLASS & FUNGSI PIPELINE
# (logika tetap di sini, tidak masuk config karena bukan "angka" tapi "cara kerja")
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


def build_preprocessing_pipeline():
    """Menyusun ColumnTransformer lengkap berdasarkan kolom-kolom di config.py"""
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


# ======================
# LOAD & SPLIT DATA
# ======================
def load_and_split_data():
    housing = pd.read_csv(config.DATA_PATH)

    # buat kategori berdasarkan median_income agar tes set sesuai
    # dengan proporsi distribusi median_income
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=config.INCOME_CAT_BINS,
        labels=config.INCOME_CAT_LABELS,
    )

    train_set, test_set = train_test_split(
        housing,
        test_size=config.TEST_SIZE,
        stratify=housing["income_cat"],
        random_state=config.RANDOM_STATE,
    )

    # hapus kolom income_cat karena sudah tidak diperlukan lagi
    for set_ in (train_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return train_set, test_set


# ======================
# TRAINING
# ======================
def train_model(housing_train_x, housing_labels_train_y):
    preprocessing = build_preprocessing_pipeline()

    full_pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "random_forest",
                RandomForestRegressor(random_state=config.RF_RANDOM_STATE),
            ),
        ]
    )

    rnd_search_forest_reg = RandomizedSearchCV(
        full_pipeline,
        param_distributions=config.PARAM_DISTRIBS,
        n_iter=config.CV_N_ITER,
        cv=config.CV_FOLDS,
        scoring=config.CV_SCORING,
        random_state=config.RANDOM_STATE,
        refit=True,
    )
    rnd_search_forest_reg.fit(housing_train_x, housing_labels_train_y)
    return rnd_search_forest_reg


# ======================
# MAIN
# ======================
def main():
    train_set, test_set = load_and_split_data()

    housing_train_x = train_set.drop(config.TARGET_COL, axis=1)
    housing_labels_train_y = train_set[config.TARGET_COL].copy()

    rnd_search_forest_reg = train_model(housing_train_x, housing_labels_train_y)

    final_model = rnd_search_forest_reg.best_estimator_
    joblib.dump(final_model, config.MODEL_OUTPUT_PATH)

    print(f"Model tersimpan di: {config.MODEL_OUTPUT_PATH}")
    print(f"Best params: {rnd_search_forest_reg.best_params_}")


if __name__ == "__main__":
    main()
