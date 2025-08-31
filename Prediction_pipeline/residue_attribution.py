import argparse
import pandas as pd
import numpy as np
import torch
import random
from captum.attr import IntegratedGradients
from itertools import groupby
import csv
from operator import itemgetter
from tqdm import tqdm
from Bio import SeqIO
import os
import torch.nn as nn
import torch.nn.functional as F


def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set dropout and batchnorm to eval mode
def set_dropout_and_bn_eval(module):
    if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d)):
        module.eval()

# ---- Mapping model names to embedding dimensions ---- #
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


# Main analysis function
def main(args):
    set_random_seeds()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Determine embedding dimension
    embedding_dim = EMBEDDING_DIM_MAP[args.embedding_type]

    # Load embeddings & sequence IDs from .npz
    if args.embedding_file.endswith(".npz"):
        data = np.load(args.embedding_file, allow_pickle=True)
        embeddings = data['embeddings']
        sequence_ids = data['sequence_ids']
    else:
        raise ValueError("Embedding file must be .npz containing 'embeddings' and 'sequence_ids'")

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)

        # Load model
    model_path = f"models/final_{args.embedding_type}_{args.model_type}.pt"

    if args.model_type == "full":
        model = EnhancedProteinModelFull(
            embedding_dim=embedding_dim, hidden_dim=128, output_dim=1, num_lstm_layers=3, num_filters=64, num_attention_heads=8, num_fc_layers=3, kernel_size=5)
    elif args.model_type == "no_cnn":
        model = EnhancedProteinModelNoCNN(embedding_dim=embedding_dim, hidden_dim=128, output_dim=1,num_lstm_layers=3, num_fc_layers=3, num_attention_heads=8)
    elif args.model_type == "no_lstm":
        model = EnhancedProteinModelNoLSTM(embedding_dim=embedding_dim, hidden_dim=128, output_dim=1,num_fc_layers=3, num_attention_heads=8, num_filters=64, kernel_size=5)
    elif args.model_type == "no_attention":
        model = EnhancedProteinModelNoAttention(embedding_dim=embedding_dim, hidden_dim=128, output_dim=1,num_lstm_layers=3, num_filters=64,num_fc_layers=3, kernel_size=5)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    ig = IntegratedGradients(model)

    # Load FASTA sequences
    fasta_sequences = list(SeqIO.parse(args.fasta_file, "fasta"))

    fake_batch_size = 16
    os.makedirs(args.output_dir, exist_ok=True)

    for idx in tqdm(range(len(fasta_sequences)), desc="Processing Sequences"):
        sequence_id = sequence_ids[idx]
        sequence = fasta_sequences[idx].seq
        seq_len = len(sequence)

        input_tensor = embeddings_tensor[idx:idx+1]
        fake_batch = input_tensor.repeat(fake_batch_size, 1, 1)

        model_mode = model.training
        try:
            model.train()
            model.apply(set_dropout_and_bn_eval)
            attributions, _ = ig.attribute(
                fake_batch,
                target=0,
                return_convergence_delta=True,
                internal_batch_size=fake_batch_size
            )
        finally:
            model.train(model_mode)

        # Only keep first "real" attribution
        attributions = attributions[0][:seq_len].detach().cpu().numpy()
        aggregated = np.sum(attributions, axis=1)
        averaged = np.mean(attributions, axis=1)

        # Save CSV
        csv_file = os.path.join(args.output_dir, f"attribution_{sequence_id}_{args.embedding_type}_{args.model_type}.csv")
        df = pd.DataFrame({
            "Residue_Index": np.arange(1, seq_len+1),
            "Residue": list(sequence),
            "Aggregated_Attribution": aggregated,
            "Averaged_Attribution": averaged
        })
        df.to_csv(csv_file, index=False)


    print("✅ Attribution analysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute raw protein attributions")
    parser.add_argument("--fasta_file", required=True, help="Input FASTA file")
    parser.add_argument("--embedding_file", required=True, help="NPZ file with embeddings and sequence_ids")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--embedding_type", required=True, choices=EMBEDDING_DIM_MAP.keys(), help="Embedding model name")
    parser.add_argument("--model_type", required=True, help="Model architecture type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)
