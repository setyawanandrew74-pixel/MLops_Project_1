# 🏠 California Housing Price Prediction — MLOps Pipeline

Project machine learning end-to-end untuk memprediksi harga rumah di California menggunakan dataset sensus 1990, dibangun dengan arsitektur pipeline modular (bukan sekadar notebook) sebagai simulasi workflow data scientist di industri real estate.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Ringkasan Project

Sebuah perusahaan properti membutuhkan sistem untuk menentukan apakah suatu daerah (district) layak untuk investasi, berdasarkan prediksi harga rumah median. Sebelumnya, estimasi dilakukan manual oleh tim ahli — lambat, mahal, dan bisa meleset lebih dari 30%.

Project ini membangun model *regression* untuk memprediksi `median_house_value` berdasarkan fitur-fitur demografis dan geografis suatu district (pendapatan, populasi, lokasi, dll), lengkap dengan pipeline modular yang bisa dijalankan ulang secara konsisten (train → evaluate → predict).

**Dataset:** [California Housing Prices](https://github.com/ageron/data) (data sensus 1990, 20.640 baris, 10 atribut)

---

## 📊 Hasil Model

Tiga algoritma dibandingkan menggunakan RMSE dan 95% bootstrap confidence interval:

| Model | RMSE | 95% CI (bootstrap) |
|---|---|---|
| Linear Regression | 70,644.16 | 65,294.62 – 87,322.01 |
| Decision Tree | 57,544.22 | 55,269.58 – 60,045.80 |
| **Random Forest (model final)** | **42,117.60** | **40,162.25 – 44,293.73** |

**Random Forest** dipilih sebagai model final karena memberikan RMSE terendah dan interval kepercayaan tersempit di antara ketiga model. Semua konfigurasi model tersedia dan dapat diganti dengan mudah melalui `config.py` bagi yang ingin bereksperimen dengan algoritma lain.

> *Catatan: metrik R² belum dihitung pada versi ini — rencana pengembangan berikutnya.*

---

## ⚙️ Arsitektur Pipeline

Project ini dirancang modular, bukan single notebook, supaya setiap tahap bisa dijalankan, diuji, dan digunakan ulang secara independen:

```
config.py                 → menyimpan konfigurasi & pilihan algoritma (Linear Regression, Decision Tree, Random Forest), versioning model dengan pathlib
utils.py                  → custom utils (ClusterSimilarity: KMeans + RBF kernel) untuk feature engineering geografis
predict_model_RND_V1.py   → melatih model sesuai konfigurasi, menyimpan model terlatih (.pkl)
evaluate.py               → menghitung RMSE dan 95% bootstrap confidence interval
train_model_RND_V1.py     → memuat model terlatih dan melakukan prediksi pada data baru
```

Alur kerja: `config.py` → `train_model_RND_V1.py` → `evaluate.py` → `predict_model_RND_V1.py`

---

## 🚀 Cara Menjalankan

```bash
# 1. Clone repository
git clone https://github.com/setyawanandrew74-pixel/MLops_Project_Multiple_Linear_Regression.git
cd MLops_Project_Multiple_Linear_Regression

# 2. Install dependencies
pip install -r requirements.txt

# 3. Latih model
python MLOps_project_1\src\train_model_RND_V1.py atau python train_model_RND_V1.py
info tambahan: file test akan tersimpan di MLOps_project_1\data\processed\test_set.csv (default)

# 4. Evaluasi model (RMSE + confidence interval)
python src/modeling/evaluate.py atau python evaluate.py

# 5. Jalankan prediksi
python src/modeling/predict_model_RND_V1.py atau python predict_model_RND_V1.py --input D:\USER\git_github\MLOps_project_1\data\processed\test_set.csv --output path simpan sesuai keinginan
catatan : untuk file predict.py wajib menggunakan parameter tambahan --input (gunakan path test_set.csv atau apapun sesuai kebutuhan) dan --output bebas
```

> Sesuaikan path di atas dengan lokasi file `train_model_RND_V1.py` / `evaluate.py` / `predict_model_RND_V1.py` yang sebenarnya di repo kamu.

---

## 📂 Struktur Project

```
├── LICENSE
├── README.md
├── requirements.txt
├── data
│   ├── external, interim, processed, raw
├── models              <- Model terlatih & hasil evaluasi
├── notebooks           <- Eksplorasi data (EDA) tahap awal
├── references
├── resource
├── reports
│   └── figures
└── src
    ├── config.py
    ├── utils.py
    ├── evaluate.py
    ├── predict_model_RND_V1.py
    ├── train_model_RND_V1.py
    
```

---

## 🔍 Insight Utama dari Eksplorasi Data

- **Lokasi adalah fitur paling berpengaruh** — district dekat laut dan di kota besar (Bay Area, LA, San Diego) cenderung punya harga lebih tinggi.
- **`median_income` berkorelasi positif kuat** dengan harga rumah, meski datanya cukup noisy.
- Ditemukan **price cap di $500,000** pada data target — berpotensi membuat model bias jika tidak ditangani.
- Fitur turunan **`bedrooms_ratio`** (rasio kamar tidur per total ruangan) terbukti lebih informatif dibanding `total_rooms` atau `total_bedrooms` mentah, dengan korelasi negatif kuat terhadap harga rumah.

*Eksplorasi data lengkap (histogram, scatter plot, peta geografis) tersedia di folder `notebooks/`.*

---

## 🛠️ Teknologi yang Digunakan

- Python
- scikit-learn (Pipeline, ColumnTransformer, RandomForestRegressor, custom Transformer)
- pandas, NumPy
- scipy.stats (bootstrap confidence interval)
- Matplotlib

---

## 📄 Lisensi

Project ini menggunakan lisensi MIT — lihat file [LICENSE](./LICENSE) untuk detail lengkap.
