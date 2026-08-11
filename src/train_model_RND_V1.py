"""
train.py
Script utama untuk training model prediksi harga rumah.
Semua path & hyperparameter dari config.py, semua class/pipeline
preprocessing dari utils.py. File ini fokus ke ALUR training saja.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

import config
from utils import preprocessing_pipeline


# ======================
# LOAD & SPLIT DATA
# ======================
def load_and_split_data():
    housing = pd.read_csv(config.DATA_PATH)

    # Buat kategori income untuk stratified sampling
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

    # Hapus kolom sementara
    for set_ in (train_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # Pastikan folder processed ada
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Simpan test set
    test_set.to_csv(config.TEST_SET_PATH, index=False)

    print(f"Test set tersimpan di : {config.TEST_SET_PATH}")

    return train_set, test_set


# ======================
# TRAIN MODEL
# ======================
def train_model(housing_train_x, housing_labels_train_y):

    preprocessing = preprocessing_pipeline()

    # Ambil model aktif dari config.py
    model = config.MODEL_CONFIG[config.ACTIVE_MODEL]["model"]
    params = config.MODEL_CONFIG[config.ACTIVE_MODEL]["params"]

    full_pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )

    search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=params,
        n_iter=config.CV_N_ITER,
        cv=config.CV_FOLDS,
        scoring=config.CV_SCORING,
        random_state=config.RANDOM_STATE,
        refit=True,
    )

    search.fit(housing_train_x, housing_labels_train_y)

    return search


# ======================
# MAIN
# ======================
def main():

    train_set, _ = load_and_split_data()

    housing_train_x = train_set.drop(config.TARGET_COLUMN, axis=1)
    housing_labels_train_y = train_set[config.TARGET_COLUMN].copy()

    search = train_model(
        housing_train_x,
        housing_labels_train_y,
    )

    final_model = search.best_estimator_

    # Pastikan folder model tersedia
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Simpan model
    joblib.dump(final_model, config.MODEL_PATH)

    print(f"\nModel berhasil disimpan di:")
    print(config.MODEL_PATH)

    print("\nModel aktif:")
    print(config.ACTIVE_MODEL)

    print("\nBest Parameters:")
    print(search.best_params_)

    print("\nBest RMSE:")
    print(-search.best_score_)


if __name__ == "__main__":
    main()
