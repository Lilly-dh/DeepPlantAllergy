import argparse
import pandas as pd
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from itertools import groupby
import csv
from operator import itemgetter
from tqdm import tqdm
from Bio import SeqIO
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.colors as mcolors  # Import for custom normalization

# Attribution smoothing function
def smooth_attributions(attributions, window_size=13):
    if len(attributions.shape) > 1:
        attributions = attributions.flatten()
    return np.convolve(attributions, np.ones(window_size) / window_size, mode='same')

# Extract motifs (with start-end)
def extract_motifs_with_positions(important_indices):
    motifs = []
    for k, g in groupby(enumerate(important_indices), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        motifs.append((group[0], group[-1] + 1))
    return motifs
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

# Set up random seed for reproducibility
def set_random_seeds():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Main analysis function
def main(args):
    set_random_seeds()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dynamically map embedding dimension based on model name
    embedding_dim = EMBEDDING_DIM_MAP[args.model_name]
    
    model_path = f"models/final_{args.model_name}.pt"
    model = EnhancedProteinModelCNNFirst(embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    ig = IntegratedGradients(model)

    fasta_sequences = list(SeqIO.parse(args.fasta_file, "fasta"))
    metadata_list = [(record.id, str(record.seq)) for record in fasta_sequences]

    predictions_df = pd.read_csv(args.prediction_file)
    pred_embeddings = np.load(args.embedding_file, allow_pickle=True)

    results_list = []
    fake_batch_size = 16

    os.makedirs(args.output_dir, exist_ok=True)

    # Attribution loop
    for batch_index in tqdm(range(len(metadata_list)), desc="Processing Sequences"):
        sequence_id = metadata_list[batch_index][0]
        sequence = fasta_sequences[batch_index].seq
        sequence_length = len(sequence)

        predicted_probability = predictions_df.loc[
            predictions_df['sequence_id'] == sequence_id, 'probability'
        ].values

        if len(predicted_probability) == 0:
            continue

        input_tensor = torch.tensor(pred_embeddings[batch_index:batch_index + 1], dtype=torch.float32).to(device)
        fake_batch = input_tensor.repeat(fake_batch_size, 1, 1)

        model_mode = model.training
        try:
            model.train()
            model.apply(set_dropout_and_bn_eval)

            # Use target=1 for the positive class (allergen)
            attributions, delta = ig.attribute(
                fake_batch,
                target=0,
                return_convergence_delta=True,
                internal_batch_size=fake_batch_size
            )
        finally:
            model.train(model_mode)

        # Use the first sample's attribution (rest are fake)
        attributions = attributions[0][:sequence_length]
        aggregated_attributions = np.sum(attributions.detach().cpu().numpy(), axis=1)
        averaged_attributions = np.mean(attributions.detach().cpu().numpy(), axis=1)

        smoothed = smooth_attributions(aggregated_attributions)
        normed = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))

        threshold = 0

        # Save attribution CSV
        csv_filename = f"{args.output_dir}/attribution_{sequence_id}_{args.model_name}.csv"
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Residue Index", "Residue", "Raw Attribution", "Mean Attribution", "Smoothed", "Normalized"])
            for i in range(sequence_length):
                writer.writerow([
                    i + 1,
                    sequence[i],
                    aggregated_attributions[i],
                    averaged_attributions[i],
                    smoothed[i],
                    normed[i]
                ])

        # Save plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 10), gridspec_kw={'height_ratios': [4, 1]})

        ax1.plot(range(sequence_length), aggregated_attributions[:sequence_length], marker='o', markersize=1, linestyle='-', color='purple', linewidth=0.5)
        ax1.set_title(f'Aggregated Importance Scores - {sequence_id}')
        ax1.set_xlabel('Residue Index')
        ax1.set_ylabel('Importance Score')
        ax1.grid(True)
        ax1.set_xticks(range(0, len(aggregated_attributions), len(aggregated_attributions) // 10))
        ax1.set_xticklabels([f'{i}' for i in range(0, len(aggregated_attributions), len(aggregated_attributions) // 10)])
        ax1.set_xlim(left=0, right=len(aggregated_attributions))
        
        # Create a custom normalization with 0 as the midpoint
        norm = mcolors.TwoSlopeNorm(vmin=aggregated_attributions.min(), vmax=aggregated_attributions.max(), vcenter=0)

        # Plot using imshow with the custom normalization
        im = ax2.imshow(aggregated_attributions[np.newaxis, :], cmap='RdGy_r', aspect='auto', norm=norm)

        ax2.set_xticks(range(sequence_length))
        ax2.set_xticklabels(sequence, rotation=0, fontsize=5)
        ax2.set_yticks([])
        ax2.set_title(f'Sequence: {sequence_id}')

        plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.2, label='Aggregated Score')
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/importance_plot_{sequence_id}_{args.model_name}.svg")
        plt.close(fig)

        # Extract motifs
        important_indices = np.where(aggregated_attributions > threshold)[0]
        motifs_with_positions = extract_motifs_with_positions(important_indices)
        motif_info = [{'motif': sequence[start:end], 'start': start+1, 'end': end} for start, end in motifs_with_positions]
        motif_sequences = [m['motif'] for m in motif_info]

        # Collect results
        results_list.append({
            'Seq_ID': sequence_id,
            'Sequence': str(sequence),
            'Length': sequence_length,
            'Predicted Probability': predicted_probability[0],
            'Motifs': motif_sequences,
            'Motif Positions': motif_info
        })

    # Save final motif summary
    summary_filename = f"{args.output_dir}/summary_attribution_{args.model_name}.csv"
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(summary_filename, index=False)
    print("✅ Attribution analysis complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Protein allergenicity prediction and attribution")
    parser.add_argument("--fasta_file", required=True, help="FASTA file with protein sequences")
    parser.add_argument("--prediction_file", required=True, help="CSV file with model predictions")
    parser.add_argument("--embedding_file", required=True, help="NPY file with embeddings")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files")
    parser.add_argument("--model_name", required=True, help="Embedding model name (e.g., 'esm', 'protbert', 'onehot', 'seqvec')")

    args = parser.parse_args()

    main(args)

