import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from src.data_utils import get_dataloaders
from models.model_build import CNN_BiLSTMWithAttention

def run_prediction():
    # 1. Load Konfigurasi
    with open("/Users/muhammadnajmirahmani/ProjectIndividu/configs/config.json", "r") as f:
        config = json.load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluasi menggunakan device: {device}")

    # 2. Persiapan Data (Hanya ambil test_loader dan scaler)
    _, test_loader, scaler, y_test_raw = get_dataloaders(
        config["data_path"], 
        config["target_col"], 
        config["seq_length"], 
        config["batch_size"]
    )

    # 3. Inisialisasi Arsitektur
    backbone = CNN_BiLSTMWithAttention(
        input_dim=config["input_dim"],
        num_filters=config["num_filters"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        dropout=config["dropout"]
    )

    # 4. Memuat Bobot (.pth) dengan Fix Prefix 'model.'
    checkpoint = torch.load(config["model_path"], map_location=device, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    backbone.load_state_dict(new_state_dict)
    
    backbone.to(device)
    backbone.eval()

    # 5. Proses Prediksi
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device).float()
            out = backbone(xb)
            all_preds.append(out.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)

    # 6. Invers Transform (Kembalikan ke harga asli)
    preds_rescaled = scaler.inverse_transform(preds)
    actual_rescaled = scaler.inverse_transform(y_test_raw.reshape(-1, 1))

    # 7. Visualisasi
    plt.figure(figsize=(12, 6))
    plt.plot(actual_rescaled, label="Harga Aktual", color="black", linestyle="--", alpha=0.6)
    plt.plot(preds_rescaled, label="Prediksi Model", color="red", linewidth=2)
    plt.title(f"Hasil Prediksi Saham: {config['target_col']}")
    plt.xlabel("Waktu (Hari)")
    plt.ylabel("Harga (IDR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Simpan hasil plot ke folder utama
    plt.savefig("hasil_prediksi.png")
    print("✅ Grafik berhasil disimpan sebagai 'hasil_prediksi.png'")
    plt.show()

if __name__ == "__main__":
    run_prediction()