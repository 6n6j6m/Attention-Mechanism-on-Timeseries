# 📈 ProjectIndividu: Stock Price Prediction (BBCA)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorch-lightning&logoColor=white)](https://www.pytorchlightning.ai/)

Proyek ini adalah studi mendalam mengenai penerapan **Attention Mechanism** pada peramalan data deret waktu (*Time-series Forecasting*) menggunakan harga penutupan saham BBCA. Fokus utama proyek ini adalah transisi dari riset eksperimental di Jupyter Notebook ke struktur kode produksi yang modular.

---

## 📂 Struktur Folder
- `src/`: Modul inti untuk *data loading*, *scaling*, dan *sequencing* (Modular).
- `models/`: Definisi arsitektur model PyTorch (`model_build.py`) dan penyimpanan bobot `.pth`.
- `data/`: Dataset historis saham BBCA (CSV).
- `notebooks/`: Kumpulan file `.ipynb` yang digunakan untuk riset awal dan *prototyping*.
- `train.py`: Script utama untuk melatih model dengan integrasi PyTorch Lightning.
- `predict.py`: Script untuk evaluasi, *inference*, dan visualisasi hasil prediksi.
- `config.json`: Pusat pengaturan parameter (Hyperparameters, Paths, & Device).

---

## 🧠 Konsep & Arsitektur
Proyek ini dibuat dengan intensi khusus untuk mempelajari bagaimana **Attention Mechanism** dapat meningkatkan kemampuan model dalam mengingat konteks jangka panjang tanpa kehilangan fokus pada volatilitas jangka pendek.

### Model Hybrid: CNN-BiLSTM with Attention
Model ini menggabungkan tiga komponen utama:
1. **CNN (1D Convolutional)**: Bertugas mengekstrak fitur lokal dan pola teknis pada time series.
2. **BiLSTM (Bidirectional LSTM)**: Menangkap ketergantungan atau dependensi temporal jangka panjang dalam data.
3. **Attention Layer**: Mekanisme yang memberikan bobot pada setiap *time step* untuk menentukan informasi mana yang paling relevan bagi prediksi harga esok hari.

**Varian Attention yang Diimplementasikan:**
- **Single-head Attention (Self Attention)**: Fokus pada satu aspek hubungan antar waktu.
- **Multi-head Attention**: Memungkinkan model untuk mempelajari berbagai proyeksi hubungan temporal secara paralel.

---

## 📊 Evaluasi Metrik
Setiap model dilatih menggunakan metode **Grid Search** untuk menemukan hyperparameter paling optimal. Kinerja model diukur menggunakan:

1. **Mean Squared Error (MSE)**:
   $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

2. **Root Mean Squared Error (RMSE)**:
   $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

---

## 💡 Struggle & Findings

Dalam proses pengerjaan proyek ini, terdapat beberapa tantangan dan temuan penting yang menjadi bahan pembelajaran utama:

### 1. Data Sparsity & Continuity
Salah satu tantangan terbesar adalah kondisi dataset yang memiliki banyak **data kosong (missing values)**. Hal ini menyebabkan urutan waktu tidak sepenuhnya kontinu, yang secara teoritis kurang memenuhi syarat ideal konsep *time-series*.

### 2. Pentingnya Analisis Statistik (EDA)
Saya menyadari adanya keterbatasan dalam proyek ini karena **melewatkan proses EDA (Exploratory Data Analysis)** serta analisis statistik *time-series* yang mendalam (seperti uji stasioneritas, musiman, atau autokorelasi). Hal ini mempengaruhi bagaimana hasil yang dihasilkan, karena jika hanya mengandalkan kompleksitas model tanpa memahami konteks dari data maka hasil yang diperoleh tidak akan optimal.

### 3. Attention Mechanism
Meskipun menghadapi tantangan data, penggunaan **Attention Mechanism** terbukti sangat membantu model dalam memahami fitur temporal. Attention Mechanism bertindak sebagai filter dinamis yang memberikan "perhatian khusus" pada langkah waktu tertentu yang memiliki signifikansi lebih tinggi terhadap target prediksi.

### 4. Konsep Q, K, V
Saya menemukan bahwa implementasi teknis mekanisme atensi mungkin akan berbeda secara signifikan pada domain yang lain (seperti Computer Vision atau NLP). Namun, konsep inti dari **Query (Q), Key (K), dan Value (V)** tetaplah terlibat dan menjadi dasar yang menghubungkan berbagai variasi arsitektur attention tersebut.
