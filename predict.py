import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv

# ---- Mapping model names to embedding dimensions ---- #
EMBEDDING_DIM_MAP = {
    "onehot": 21,
    "protbert": 1024,
    "seqvec": 1024,
    "esm": 1280
}

# ---- Define the attention block ---- #
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size), "Embed size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

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
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)
        return self.fc_out(out)

# ---- Define the full model architecture ---- #
class EnhancedProteinModelCNNFirst(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=512, output_dim=1, num_lstm_layers=3, num_fc_layers=2, num_attention_heads=8, num_filters=128, kernel_size=5):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.batch_norm_conv = nn.BatchNorm1d(num_filters)
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_dim, num_layers=num_lstm_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.attention = SelfAttention(embed_size=hidden_dim * 2, heads=num_attention_heads)
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
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.batch_norm_conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.attention(x, x, x)
        x = self.pool(x.permute(0, 2, 1)).squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        for fc, bn in zip(self.fc_layers, self.batch_norm_fc):
            x = self.relu(bn(fc(x)))
            x = self.dropout(x)
        return self.fc_output(x)

# ---- Prediction pipeline ---- #
def predict(model_name, input_emb_path, input_ids_path, output_csv="predictions.csv", batch_size=5):
    if model_name not in EMBEDDING_DIM_MAP:
        raise ValueError(f"Invalid model name '{model_name}'. Choose from: {list(EMBEDDING_DIM_MAP.keys())}")

    embedding_dim = EMBEDDING_DIM_MAP[model_name]
    model_path = f"models/final_{model_name}.pt"

    # Load model
    model = EnhancedProteinModelCNNFirst(embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Load data
    embeddings = np.load(input_emb_path, allow_pickle=True)
    sequence_ids = np.load(input_ids_path, allow_pickle=True)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    prediction_probs = []

    for i in range(0, len(embeddings_tensor), batch_size):
        batch = embeddings_tensor[i:i + batch_size]
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
            prediction_probs.extend(probs)

    # Save predictions
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sequence_id', 'probability', 'prediction', 'comment'])
        for seq_id, prob_array in zip(sequence_ids, prediction_probs):
            prob = prob_array[0]
            if prob < 0.5:
                prediction, comment = 0, "probably not allergen"
            elif prob < 0.8:
                prediction, comment = 1, "potentially allergen"
            else:
                prediction, comment = 1, "high probability allergen"
            writer.writerow([seq_id, f"{prob:.4f}", prediction, comment])
    print(f"✅ Predictions saved as {output_csv}")

# ---- Command-line interface ---- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run allergenicity prediction with specified model.")
    parser.add_argument("--model", required=True, help="Model name (onehot, protbert, seqvec, esm)")
    parser.add_argument("--input_emb", required=True, help="Path to .npy embeddings file")
    parser.add_argument("--input_ids", required=True, help="Path to .npy sequence IDs file")
    args = parser.parse_args()

    # Always auto-generate output filename based on model name
    output_csv = f"{args.model}_prediction_results.csv"

    predict(args.model, args.input_emb, args.input_ids, output_csv)

