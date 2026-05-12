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

# Sekilas Tentang Struktur Data
<img width="5118" height="1207" alt="struktur_data_housing" src="https://github.com/user-attachments/assets/66224a3c-21dd-442a-9caa-391420eb4b46" />
Di awal, Saya pakai metode head(). Hasilnya Gambar sebuah tabel yang menampilkan 5 baris pertama dari dataset.
- Setiap baris mewakili satu distrik (daerah).
- Ada 10 atribut (kolom):

Metode info() memberi saya ringkasan data:
<img width="465" height="388" alt="housing_info" src="https://github.com/user-attachments/assets/8cfe33a0-9937-4a3e-a4a3-7edd8416ed5f" />
1. Jumlah total baris (instances): 20.640
(Cukup kecil menurut standar machine learning, tapi bagus untuk belajar)
2.Jumlah nilai yang tidak kosong per kolom:
Contoh: total_bedrooms hanya punya 20.433 nilai non-null → berarti 207 distrik kehilangan data atribut ini (nanti perlu ditangani).
3.Tipe data:
Semua atribut numerik kecuali ocean_proximity yang tipe-nya object → ini teks (kategorikal).
<img width="343" height="203" alt="Ocean_Struktur_variabel" src="https://github.com/user-attachments/assets/afed08e2-0bc7-4e2a-bdd2-887837cc23c0" />

# Ringkasan Statistik Tentang Data
Saya menggunakan metode describe() yang menghasilkan sebuah tabel statistik untuk setiap atribut numerik.
<img width="1455" height="1446" alt="statistik_housing" src="https://github.com/user-attachments/assets/7c6b29cd-52d5-46d1-97cf-5657771edb2c" />

sedikit info untuk fitur housing_median_age(rata-rata usia rumah):
1.Dari keseluruhan data didapatkan sebanyak 25% distrik berumur kurang dari 18 tahun
2.Dan Hampir setengah data 50% (median) berusia kurang dari 29 tahun
3.75% < 37 tahun

# Histogram Distribusi Data
<img width="1009" height="990" alt="histogram_housing" src="https://github.com/user-attachments/assets/48b27658-0f64-4c3e-ab45-6645e7cff94d" />
Pengamatan Penting dari Histogram
a. Atribut median_income (pendapatan median)
-Tidak dalam dolar asli, sudah diubah skala dan dibatasi (capped).
-Nilai maksimal 15 (sebenarnya 15.0001), nilai minimal 0,5 (sebenarnya 0,4999).
-Angka 3 artinya sekitar $30.000 (karena dikali 10.000).
Ini hal biasa dalam ML, tapi kita harus tahu bagaimana data diproses.

b. housing_median_age dan median_house_value juga dibatasi
-Khusus median_house_value (target/label kita), pembatasan ini bisa jadi masalah serius.
-Jika sistem diminta memprediksi harga di atas batas maksimal, ia tidak bisa belajar dari data yang ada.
Solusi jika perlu prediksi di atas batas:
-Kumpulkan data label yang benar untuk distrik yang dibatasi.
-Hapus distrik tersebut dari data latih dan data uji.

c. Skala yang berbeda-beda
-Setiap atribut punya rentang nilai berbeda (misal median_income 0–15, total_rooms bisa sampai ribuan). Ini perlu ditangani nanti (feature scaling).

d. Kemencengan ke kanan (right-skewed)
Banyak histogram yang ekornya panjang ke kanan. Artinya:
-Sebagian besar data terpusat di kiri (nilai kecil)
-Hanya sedikit data yang sangat besar (di kanan)
Contoh: Pendapatan(median income) kebanyakan orang di kisaran menengah ke bawah, tapi ada sedikit yang super kaya → grafik menceng ke kanan.

Dampak: Beberapa algoritma ML sulit mendeteksi pola. Nanti kita akan mengubah bentuk distribusi menjadi lebih simetris (seperti lonceng).

# Ringkasan Akhir Untuk Struktur Data
Dengan melihat head(), info(), describe(), dan histogram, saya sekarang lebih memahami data yang akan saya olah. saya tahu:
1. Ada data kosong (207 total_bedrooms hilang)
2. Ada atribut teks (ocean_proximity)
3. Beberapa atribut dibatasi (capped) – perlu perhatian khusus pada target
4. Skala berbeda dan distribusi menceng – perlu transformasi nanti

## Mmmbuat set data uji dan latih
Sebelum menganalisis data lebih jauh, saya  membuat:
80% data → latih
20% data → uji
Para ahli dari pengloah dataset ini mengatakan:
variabel: median_income
sangat penting untuk memprediksi harga rumah.

Maka saya membuat data uji dan latih harus memiliki distribusi income yang mirip dataset asli(sesuai dengan ditribusi di dataset).
begini lah grafik distribusinya : 
<img width="580" height="432" alt="Distirubusi Data income" src="https://github.com/user-attachments/assets/d91a89c7-0241-4acd-bb6a-b10fbc03684f" />
Sumbu X: kategori income (1–5)
Sumbu Y: jumlah district
Isi grafik menunjukkan: mayoritas income berada di kategori tengah income sangat tinggi jumlahnya lebih sedikit
Dan ini presentase proporsi data latih dan data uji yang sudah proses:
<img width="400" height="143" alt="Proporsi_train_test_set" src="https://github.com/user-attachments/assets/5166765b-18d5-43f8-91b9-98a8be4f6009" />

# EDA(Eksplorasi Data Analysis)
Di bagian ini saya mulai melakukan: visualisasi data geografis
karena dataset memiliki informasi lokasi:
longitude (garis bujur)
latitude (garis lintang)
Tujuannya: melihat pola persebaran rumah langsung dari gambar/map.
<img width="834" height="607" alt="Eksplorasi geografis data housing" src="https://github.com/user-attachments/assets/f81ecdc4-5f43-4fbc-b63e-2b6ca956df5d" />
Setiap titik menunjukkan:
| Visual           | Makna            |
| ---------------- | ---------------- |
| Posisi titik     | lokasi geografis |
| Ukuran lingkaran | jumlah populasi  |
| Warna            | harga rumah      |
merah = rumah mahal
biru = rumah murah
lingkaran besar = populasi besar

gambar tersebut menunjukkan: titik-titik lokasi district di California.
Karena koordinat geografis dipakai, bentuknya terlihat seperti peta California
sekarang terlihat:
Bay Area
Los Angeles
San Diego
Central Valley
sebagai area dengan kepadatan tinggi.

**Insight yang Didapat**
harga rumah sangat dipengaruhi lokasi.

Contohnya:
-dekat laut → lebih mahal
-kota besar → lebih mahal
-daerah padat → sering lebih mahal
Sekarang saya tahu: lokasi adalah fitur penting.

Artinya model machine learning nanti kemungkinan besar akan belajar bahwa:
-latitude,
-longitude,
punya pengaruh besar terhadap harga rumah.

-ocean proximity
fitur “dekat laut(ocean proximity)” kemungkinan penting.
Tetapi:
-tidak selalu benar,
-ada area pantai yang tidak mahal.

Jadi hubungan fitur dengan harga:
-tidak selalu sesederhana “dekat laut = mahal”.


Dan juga sepertinya algoritma clustering bisa dipakai untuk:
-menemukan area-area penting,
-mengelompokkan wilayah mirip.

Di tahap ini saya belajar bahwa:
visualisasi membantu menemukan pola penting pada data, Dan dari grafik saya menemukan:
-lokasi memengaruhi harga rumah,
-kepadatan populasi penting,
-daerah tertentu membentuk cluster,
-fitur geografis kemungkinan sangat berguna untuk model ML

Bagian ini saya sedang membahas tentang hubungan antar fitur (correlation) dalam dataset housing 🏠📊
berikut laporan tabel fitur-fitur terhadap fitur median_house_value
<img width="326" height="310" alt="fitur median income" src="https://github.com/user-attachments/assets/2cecdaee-0a59-47e2-bc19-2ba997e733fe" />

Namun, Karena angka korelasi kadang kurang representatif, kita juga perlu visualisasi.
berikut visualisasi korelasi beberapa fitur yang sudah di buat:
<img width="1021" height="707" alt="scatter matrix beberapa fitur data housing" src="https://github.com/user-attachments/assets/43caa310-f261-4050-af81-5702461684b5" />

Dari hasil sementara ini fitur paling menjanjikan adalah median_income
Maka saya buat scatter plot khusus:
<img width="598" height="433" alt="scatter plot median income dan median house value" src="https://github.com/user-attachments/assets/c648272f-1d83-4798-be58-831fb17fb234" />

ini menunjukkan:
a) Ada tren naik
-Semakin tinggi income,
-harga rumah cenderung naik.
Artinya korelasi positif kuat.

b) Tapi datanya noisy
“Noisy” artinya:
-tidak rapi
-titiknya menyebar
-ada variasi acak
Karena dunia nyata memang tidak sempurna.
Contoh:
-dua daerah income sama tapi harga rumah berbeda karena faktor lain.

Price Cap yang Terlihat
Ada garis horizontal di:
-$500,000
Artinya:
-banyak data berhenti di angka itu.
-kemungkinan dataset dibatasi maksimal 500k.

Ini bisa jadi Masalah
Model machine learning bisa belajar:
-“Oh, harga maksimal selalu 500k.”
Padahal di dunia nyata bisa lebih mahal, Akibatnya model jadi bias.

Saya juga melihat garis di sekitar:
-450k
-350k
-280k

Ini mencurigakan karena:
-terlihat terlalu rapi
-seperti hasil pembulatan atau batas buatan dataset.
Biasanya disebut: data quirks / keanehan data.


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
