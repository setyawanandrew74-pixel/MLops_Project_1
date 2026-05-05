# Gambaran Untuk Project Ini
Kali ini saya membahas tentang proses lengkap (end-to-end) dalam membuat project machine learning.

Saya membayangkan diterima kerja sebagai data scientist di perusahaan properti (real estate). Dan saya dikasih tugas untuk bikin sistem berbasis machine learning.

👉 Tujuan project ini bukan belajar bisnis properti, tapi untuk belajar mengetahui alur kerja ML dari awal sampai akhir.

🏠 Dataset yang Dipakai di kali Ini
Dataset yang digunakan adalah: 👉 California Housing Prices dataset source= https://github.com/ageron/data

📌 Isinya: Data sensus tahun 1990 di California Digunakan untuk prediksi harga rumah
📌 Catatan penting:Data ini cukup lama Tapi tetap dipakai karena: Mudah dipahami Cocok untuk belajar
📌 Penyesuaian: Ditambah 1 fitur kategori Beberapa fitur dihapus (biar lebih simpel untuk pembelajaran) isinya seperti:
Jumlah penduduk
Pendapatan rata-rata
Harga rumah rata-rata
dll

👉 Unit datanya disebut block group
➡️ Supaya gampang, disebut district (daerah kecil)
📌 Satu district kira-kira berisi: 600 – 3.000 orang

🎯 Tujuan Model
Model saya harus bisa: Memprediksi harga rumah di suatu district berdasarkan data lainnya

📌 Contoh:
Input: pendapatan, populasi, dll
Output: harga rumah

💼 Jawaban dari Bos
Model kamu akan dipakai oleh sistem lain untuk: 👉 Menentukan apakah suatu daerah layak untuk investasi atau tidak

💰 Artinya:
Kalau prediksi saya salah → bisa rugi besar
Kalau benar → perusahaan untung

Yang di Lakukan Perusahaan Saat Ini (Manual) 
Sekarang mereka pakai cara lama:
👨‍💼 Tim ahli: Kumpulin data, Estimasi harga pakai aturan rumit
❌ Masalah: Lama, Mahal dan Tidak akurat (bisa meleset > 30%)
👉 Makanya mereka butuh machine learning

Menentukan Jenis Machine Learning
Sekarang saya klasifikasikan masalahnya.

✅ Apakah Supervised atau Tidak? 👉 Supervised Learning
Kenapa? Karena:
Ada input (data district)
Ada label (harga rumah yang sudah diketahui)

✅ Classification atau Regression? 👉 Regression 
Karena: Output berupa angka (harga rumah) Bukan kategori

✅ Multiple Regression
Karena: Inputnya banyak:
populasi
income
dll

✅ Univariate Regression 
Karena: Output cuma 1:
harga rumah

| Jenis             | Jawaban |
| ----------------- | ------- |
| Supervised        | ✅       |
| Regression        | ✅       |
| Multiple features | ✅       |
| Single output     | ✅       |

Batch Learning vs Online Learning
👉 Batch Learning 
Kenapa? Data tidak berubah terus-menerus, Tidak real-time, Data cukup kecil (muat di memory)


to be continue


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── src                         <- Source code for this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    │    
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations 
    │
    └── services                <- Service classes to connect with external platforms, tools, or APIs
        └── __init__.py 
```
