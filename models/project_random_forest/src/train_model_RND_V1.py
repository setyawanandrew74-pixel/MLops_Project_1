"""
train.py
Script utama untuk training model prediksi harga rumah.
Semua path & hyperparameter dari config.py, semua class/pipeline
preprocessing dari transformers.py. File ini fokus ke ALUR training saja.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

import config
from utils import preprocessing_pipeline


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

    # simpan test_set ke CSV supaya bisa dipakai ulang oleh predict.py
    # atau evaluate.py, tanpa perlu split ulang dari housing.csv
    os.makedirs(os.path.dirname(config.TEST_SET_OUTPUT_PATH), exist_ok=True)
    test_set.to_csv(config.TEST_SET_OUTPUT_PATH, index=False)
    print(f"Test set tersimpan di: {config.TEST_SET_OUTPUT_PATH}")

    return train_set, test_set


# ======================
# TRAINING
# ======================
def train_model(housing_train_x, housing_labels_train_y):
    preprocessing = preprocessing_pipeline()

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
    print(f"Best score: {-rnd_search_forest_reg.best_score_}")


if __name__ == "__main__":
    main()
