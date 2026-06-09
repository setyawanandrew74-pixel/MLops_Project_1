import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

housing = pd.read_csv("D:\\USER\\git_github\\MLOps_project_1\\data\\raw\\housing.csv")

# buat kategori berdasarkan median_income agar tes set sesuai dengan proporsi ditribusi median_income
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

# buat data train dan test dengan stratified sampling berdasarkan income_cat
train_set, test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42
)

# cek hasil proporsi income_cat di test set dan train set
test_set["income_cat"].value_counts() / len(test_set)
train_set["income_cat"].value_counts() / len(train_set)


# hapus kolom income_cat karena sudah tidak diperlukan lagi
for set_ in (train_set, test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = train_set.copy()
housing_train = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()


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


log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler(),
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
)

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)

default_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

preprocessing = ColumnTransformer(
    [
        ("bedrooms_per_room", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_household", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_household", ratio_pipeline(), ["population", "households"]),
        (
            "log",
            log_pipeline,
            [
                "total_bedrooms",
                "total_rooms",
                "households",
                "population",
                "median_income",
            ],
        ),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_pipeline,
)
housing_prepared = preprocessing.fit_transform(housing_train)


print(housing_prepared.shape)
