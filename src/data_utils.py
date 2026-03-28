import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset(Dataset):
    """
    Dataset class untuk data time series.
    Mengubah data yang sudah di-sequence menjadi tensor PyTorch.
    """
    def __init__(self, x, y):
        # Pastikan data dalam format float32 untuk stabilitas di MPS (M5)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_sequences(data, seq_length):
    """
    Fungsi windowing untuk membuat pasangan input (X) dan target (y).
    Misal: seq_length=30, maka X adalah hari 1-30, y adalah hari ke-31.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : (i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_dataloaders(file_path, target_col, seq_length, batch_size, train_split=0.8):
    """
    Fungsi utama untuk memproses CSV hingga menjadi DataLoader.
    """
    # 1. Load Data
    df = pd.read_csv(file_path)
    # Ambil kolom target (misal: 'Close' atau 'Value')
    data = df[target_col].values.reshape(-1, 1)

    # 2. Scaling (0-1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # 3. Create Sequences
    X, y = create_sequences(data_scaled, seq_length)

    # 4. Split Train & Test
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 5. Buat Dataset & DataLoader
    train_ds = TimeSeriesDataset(X_train, y_train)
    test_ds = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)  # Tidak shuffle untuk data time series
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler, y_test