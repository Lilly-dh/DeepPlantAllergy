import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv
from datetime import datetime
import random

# ---- Embedding dimensions ---- #
EMBEDDING_DIM_MAP = {
    "onehot": 21,
    "protbert": 1024,
    "seqvec": 1024,
    "esm": 1280
}

# ---- Self-Attention Block ---- #
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.attention_weights = None

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(energy / (self.embed_size ** 0.5), dim=3)
        self.attention_weights = attention
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)
        return self.fc_out(out)


# 1. Full model: CNN + LSTM + Attention
class EnhancedProteinModelFull(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_lstm_layers=3, num_fc_layers=3, num_attention_heads=8,
                 num_filters=64, kernel_size=5):
        super().__init__()
        
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                                kernel_size=kernel_size, padding=kernel_size // 2)
        self.batch_norm_conv = nn.BatchNorm1d(num_filters)
        
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_dim,
                            num_layers=num_lstm_layers, batch_first=True,
                            dropout=0.5, bidirectional=True)
        
        self.attention = SelfAttention(embed_size=hidden_dim * 2,
                                       heads=num_attention_heads)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc_layers = nn.ModuleList()
        self.batch_norm_fc = nn.ModuleList()
        input_dim = hidden_dim * 2
        
        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norm_fc.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.batch_norm_conv(x)
        x = x.permute(0, 2, 1)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Attention
        x = self.attention(x, x, x)
        
        # Pooling
        x = self.pool(x.permute(0, 2, 1)).squeeze()
        
        # Fully connected
        for fc, bn in zip(self.fc_layers, self.batch_norm_fc):
            x = self.relu(bn(fc(x)))
            x = self.dropout(x)
        
        x = self.fc_output(x)
        return x


# 2. No CNN: LSTM + Attention
class EnhancedProteinModelNoCNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_lstm_layers=3, num_fc_layers=3, num_attention_heads=8):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_lstm_layers, batch_first=True,
                            dropout=0.5, bidirectional=True)
        
        self.attention = SelfAttention(embed_size=hidden_dim * 2,
                                       heads=num_attention_heads)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc_layers = nn.ModuleList()
        self.batch_norm_fc = nn.ModuleList()
        input_dim = hidden_dim * 2
        
        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norm_fc.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # LSTM
        x, _ = self.lstm(x)
        
        # Attention
        x = self.attention(x, x, x)
        
        # Pooling
        x = self.pool(x.permute(0, 2, 1)).squeeze()
        
        # Fully connected
        for fc, bn in zip(self.fc_layers, self.batch_norm_fc):
            x = self.relu(bn(fc(x)))
            x = self.dropout(x)
        
        x = self.fc_output(x)
        return x


# 3. No LSTM: CNN + Attention
class EnhancedProteinModelNoLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_fc_layers=3, num_attention_heads=8,
                 num_filters=64, kernel_size=5):
        super().__init__()
        
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                                kernel_size=kernel_size, padding=kernel_size // 2)
        self.batch_norm_conv = nn.BatchNorm1d(num_filters)
        
        self.attention = SelfAttention(embed_size=num_filters,
                                       heads=num_attention_heads)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc_layers = nn.ModuleList()
        self.batch_norm_fc = nn.ModuleList()
        input_dim = num_filters
        
        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norm_fc.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.batch_norm_conv(x)
        x = x.permute(0, 2, 1)
        
        # Attention
        x = self.attention(x, x, x)
        
        # Pooling
        x = self.pool(x.permute(0, 2, 1)).squeeze()
        
        # Fully connected
        for fc, bn in zip(self.fc_layers, self.batch_norm_fc):
            x = self.relu(bn(fc(x)))
            x = self.dropout(x)
        
        x = self.fc_output(x)
        return x


# 4. No Attention: CNN + LSTM
class EnhancedProteinModelNoAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_lstm_layers=3, num_fc_layers=3,
                 num_filters=64, kernel_size=5):
        super().__init__()
        
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                                kernel_size=kernel_size, padding=kernel_size // 2)
        self.batch_norm_conv = nn.BatchNorm1d(num_filters)
        
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_dim,
                            num_layers=num_lstm_layers, batch_first=True,
                            dropout=0.5, bidirectional=True)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc_layers = nn.ModuleList()
        self.batch_norm_fc = nn.ModuleList()
        input_dim = hidden_dim * 2
        
        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norm_fc.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.batch_norm_conv(x)
        x = x.permute(0, 2, 1)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Pooling
        x = self.pool(x.permute(0, 2, 1)).squeeze()
        
        # Fully connected
        for fc, bn in zip(self.fc_layers, self.batch_norm_fc):
            x = self.relu(bn(fc(x)))
            x = self.dropout(x)
        
        x = self.fc_output(x)
        return x


# ---- Prediction Function ---- #
def predict(embedding_type, model_type, input_path, output_csv=None, batch_size=16, seed=42):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Determine embedding dimension
    embedding_dim = EMBEDDING_DIM_MAP[embedding_type]

    # Load embeddings & IDs
    if input_path.endswith(".npz"):
        data = np.load(input_path)
        embeddings = data['embeddings']
        sequence_ids = data['sequence_ids']
    else:  # assume .npy
        embeddings = np.load(input_path)
        raise ValueError("For .npy embeddings, sequence IDs must be provided in a .npy or separately.")

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Load model
    model_path = f"models/final_{embedding_type}_{model_type}.pt"

    if model_type == "full":
        model = EnhancedProteinModelFull(
            embedding_dim=embedding_dim, hidden_dim=128, output_dim=1, num_lstm_layers=3, num_filters=64, num_attention_heads=8, num_fc_layers=3, kernel_size=5)
    elif model_type == "no_cnn":
        model = EnhancedProteinModelNoCNN(embedding_dim=embedding_dim, hidden_dim=128, output_dim=1,num_lstm_layers=3, num_fc_layers=3, num_attention_heads=8)
    elif model_type == "no_lstm":
        model = EnhancedProteinModelNoLSTM(embedding_dim=embedding_dim, hidden_dim=128, output_dim=1,num_fc_layers=3, num_attention_heads=8, num_filters=64, kernel_size=5)
    elif model_type == "no_attention":
        model = EnhancedProteinModelNoAttention(embedding_dim=embedding_dim, hidden_dim=128, output_dim=1,num_lstm_layers=3, num_filters=64,num_fc_layers=3, kernel_size=5)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Predictions
    prediction_probs = []
    for i in range(0, len(embeddings_tensor), batch_size):
        batch = embeddings_tensor[i:i + batch_size]
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
            prediction_probs.extend(probs)

    # Output CSV
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_csv = f"{embedding_type}_{model_type}_predictions_{timestamp}.csv"

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence_id', 'probability', 'prediction', 'comment'])
        for seq_id, prob_array in zip(sequence_ids, prediction_probs):
            prob = prob_array[0]
            if prob < 0.5:
                pred, comment = 0, "probably not allergen"
            elif prob < 0.8:
                pred, comment = 1, "potentially allergen"
            else:
                pred, comment = 1, "high probability allergen"
            writer.writerow([seq_id, f"{prob:.4f}", pred, comment])

    print(f"✅ Predictions saved to {output_csv}")

# ---- CLI ---- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allergenicity prediction")
    parser.add_argument("--embedding", required=True,
                        choices=EMBEDDING_DIM_MAP.keys(), help="Embedding type")
    parser.add_argument("--model_type", required=True,
                        choices=["full", "no_cnn", "no_lstm", "no_attention"], help="Model architecture")
    parser.add_argument("--input", required=True, help="Path to embeddings (.npz)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    predict(args.embedding, args.model_type, args.input, batch_size=args.batch_size, seed=args.seed)
