# 📈 ProjectIndividu: Stock Price Prediction (BBCA)
Proyek ini menggunakan arsitektur hybrid **CNN-BiLSTM with Attention** untuk memprediksi harga penutupan saham BBCA. Dioptimalkan khusus untuk menjalankan akselerasi **MPS (Metal Performance Shaders)** pada chip Apple Silicon (M5).

---

## 📂 Struktur Folder
- `src/`: Modul inti untuk data loading & preprocessing.
- `models/`: Arsitektur model PyTorch & penyimpanan bobot `.pth`.
- `data/`: Dataset historis saham (CSV).
- `train.py`: Script untuk melatih model.
- `predict.py`: Script untuk evaluasi & visualisasi hasil.
- `config.json`: Pusat pengaturan parameter & path.
- ipynb : Tempat eksperimen
