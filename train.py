import lightning as L
import torch
import os
from src.data_utils import get_dataloaders
from models.model_build import TimeSeriesModel, CNN_BiLSTMWithAttention

def train():
    data_path = "/Users/muhammadnajmirahmani/ProjectIndividu/data/bbca_closing_price_22_to_23.csv"
    
    train_loader, val_loader, scaler, y_test = get_dataloaders(
        data_path, 
        "Close", 
        seq_length=30, 
        batch_size=64
    )

    config = {
        "input_dim": 1,
        "output_dim": 1,
        "num_filters": 32,
        "hidden_dim": 100,
        "dropout": 0.3
    }
    
    # Inisialisasi arsitektur (Backbone)
    backbone = CNN_BiLSTMWithAttention(**config)
    
    path = "/Users/muhammadnajmirahmani/ProjectIndividu/models/CNN_BiLSTMWithAttention_best_model.pth"
    
    if os.path.exists(path):
        # Load checkpoint mentah
        checkpoint = torch.load(path, map_location="mps", weights_only=True)
        
        # Cek jika file .pth berisi dictionary 'state_dict'
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Hapus prefix 'model.' agar cocok dengan nama layer di backbone
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("model.", "")  # Menghapus 'model.'
            new_state_dict[name] = v
        
        # Masukkan bobot yang sudah dibersihkan ke backbone
        backbone.load_state_dict(new_state_dict)

    # Bungkus backbone ke dalam LightningModule
    model = TimeSeriesModel(model=backbone, learning_rate=1e-3)

    # 3. Trainer
    # Kita set log_every_n_steps agar tidak terlalu berisik di terminal
    trainer = L.Trainer(
        accelerator="mps", 
        devices=1, 
        max_epochs=100,
        log_every_n_steps=10
    )
    
    # Jalankan Training
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train()