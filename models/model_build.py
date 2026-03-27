import lightning as L
import torch
import torch.nn as nn

# Get scaled dot-product attention from torch.nn.functional and MultiheadAttention from torch.nn
from torch.nn.functional import scaled_dot_product_attention
from torch.nn import MultiheadAttention
from torch import nn

# Get self attention single head from torch.nn
class SelfAttentionSingleHead(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttentionSingleHead, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_output = scaled_dot_product_attention(Q, K, V)
        return attn_output

class SelfAttentionMultiHead(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionMultiHead, self).__init__()
        self.multihead_attn = MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output

class BiLSTMWithMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = SelfAttentionMultiHead(hidden_dim * 2, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                 # (B, T, 2H)
        attn_out = self.attention(lstm_out)        # (B, T, 2H)
        attn_out = self.dropout(attn_out)
        # Context vector formed by weighted sum of all time steps
        context_vector = torch.mean(attn_out, dim=1)

        out = self.fc(context_vector)              # (B, output_dim)
        return out

class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = SelfAttentionSingleHead(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                 # (B, T, H)
        attn_out = self.attention(lstm_out)        # (B, T, H)
        attn_out = self.dropout(attn_out)
        # Context vector formed by weighted sum of all time steps
        context_vector = torch.mean(attn_out, dim=1)  # (B, H)
        out = self.fc(context_vector)              # (B, output_dim)
        return out

class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = SelfAttentionSingleHead(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                 # (B, T, 2H)
        attn_out = self.attention(lstm_out)        # (B, T, 2H)
        attn_out = self.dropout(attn_out)
        # Context vector formed by weighted sum of all time steps
        context_vector = torch.mean(attn_out, dim=1)  # (B, 2H)
        out = self.fc(context_vector)              # (B, output_dim)
        return out

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                 # (B, T, 2H)
        lstm_out = self.dropout(lstm_out)
        # Context vector formed by weighted sum of all time steps
        context_vector = torch.mean(lstm_out, dim=1)  # (B, 2H)
        out = self.fc(context_vector)              # (B, output_dim)
        return out


class CNN(nn.Module):
    def __init__(self, input_dim, num_filters, output_dim, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3)        # removes hard-coded sequence length dependency
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()        # (B, input_dim, T)
        conv_out = torch.relu(self.conv1(x))       # (B, F, T')
        pooled = self.pool(conv_out)           # (B, F, T'')
        global_max, _ = torch.max(pooled, dim=2)  # (B, F)
        global_max = self.dropout(global_max)
        out = self.fc(global_max)                      # (B, output_dim)
        return out

class CNNWithAttention(nn.Module):
    def __init__(self, input_dim, num_filters, output_dim, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.attention = SelfAttentionSingleHead(num_filters)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()        # (B, input_dim, T)
        conv_out = torch.relu(self.conv1(x))       # (B, F, T')
        pooled_out = self.pool(conv_out)           # (B, F, T'')
        pooled_out = pooled_out.permute(0, 2, 1).contiguous()  # (B, T'', F)
        attn_out = self.attention(pooled_out)      # (B, T'', F)
        attn_out = self.dropout(attn_out)
        # Context vector formed by weighted sum of all time steps
        context_vector = torch.mean(attn_out, dim=1)  # (B, F)
        out = self.fc(context_vector)              # (B, output_dim)
        return out


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, num_filters, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(num_filters, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()        # (B, input_dim, T)
        conv_out = torch.relu(self.conv1(x))       # (B, F, T')
        pooled_out = self.pool(conv_out)           # (B, F, T'')
        pooled_out = pooled_out.permute(0, 2, 1).contiguous()  # (B, T'', F)
        lstm_out, _ = self.lstm(pooled_out)        # (B, T'', 2H)
        lstm_out = self.dropout(lstm_out)
        # Context vector formed by weighted sum of all time steps
        context_vector = torch.mean(lstm_out, dim=1)  # (B, 2H)
        out = self.fc(context_vector)              # (B, output_dim)
        return out

class CNN_BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, num_filters, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(num_filters, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = SelfAttentionSingleHead(hidden_dim * 2)  # Using single-head attention for simplicity
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()        # (B, input_dim, T)
        conv_out = torch.relu(self.conv1(x))       # (B, F, T')
        pooled_out = self.pool(conv_out)           # (B, F, T'')
        pooled_out = pooled_out.permute(0, 2, 1).contiguous()  # (B, T'', F)
        lstm_out, _ = self.lstm(pooled_out)        # (B, T'', 2H)
        attn_out = self.attention(lstm_out)        # (B, T'', 2H)
        attn_out = self.dropout(attn_out)
        # Context vector formed by weighted sum of all time steps
        context_vector = torch.mean(attn_out, dim=1)  # (B, 2H)
        out = self.fc(context_vector)              # (B, output_dim)
        return out

class TimeSeriesModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
