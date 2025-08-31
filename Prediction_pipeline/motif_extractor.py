#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby
from operator import itemgetter
import os

# -------------------
# Smoothing function
# -------------------
def smooth_attributions(attributions, window_size):
    return np.convolve(attributions, np.ones(window_size)/window_size, mode='same')

# -------------------
# Motif extraction
# -------------------
def extract_motifs_with_positions(indices, smoothed, sequence, min_len=3):
    motifs = []
    for k, g in groupby(enumerate(indices), lambda x: x[0]-x[1]):
        group = list(map(itemgetter(1), g))
        if (group[-1] - group[0] + 1) >= min_len:
            start, end = group[0], group[-1]+1  # Python slice end+1
            peptide = "".join(sequence[start:end])
            avg_attr = float(np.mean(smoothed[start:end]))
            motifs.append((start+1, end, peptide, avg_attr))  # 1-based indexing
    return motifs

# -------------------
# Threshold calculation
# -------------------
def compute_threshold(smoothed, method='percentile', percentile=80):
    if method == 'midpoint':
        return 0.5 * (np.min(smoothed) + np.max(smoothed))
    elif method == 'percentile':
        return np.percentile(smoothed, percentile)
    else:
        raise ValueError(f"Unknown threshold method: {method}")

# -------------------
# Main
# -------------------
def main():
    parser = argparse.ArgumentParser(description="Motif extraction and plotting for one embedding CSV (multiple windows)")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file")
    parser.add_argument("--windows", type=int, nargs='+', default=[5], help="List of smoothing window sizes")
    parser.add_argument("--threshold", type=str, choices=['percentile', 'midpoint'], default='percentile', help="Threshold method")
    parser.add_argument("--percentile", type=int, default=80, help="Percentile cutoff if using percentile threshold")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length")
    parser.add_argument("--min-len", type=int, default=3, help="Minimum motif length")
    parser.add_argument("--output", type=str, default="motifs_output.csv", help="Output CSV filename for motifs")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)
    raw_attr = df['Raw Attribution'].astype(np.float64).values
    sequence = df['Residue'].astype(str).str.strip().values

    # Extract embedding name and model name from filename
    base = os.path.basename(args.csv).replace(".csv", "")
    parts = base.split("_")
    embedding_name = parts[4] if len(parts) > 4 else "unknown"

    # Detect model name
    model_names = ["full", "no_cnn", "no_lstm", "no_attention"]
    model_name = "unknown"
    for name in model_names:
        if name in base.lower():
            model_name = name
            break

    all_motifs = []

    for w in args.windows:
        # Smooth and threshold
        smoothed = smooth_attributions(raw_attr, window_size=w)
        if args.threshold == "percentile":
            threshold = compute_threshold(smoothed, method="percentile", percentile=args.percentile)
        else:  # midpoint
            threshold = compute_threshold(smoothed, method="midpoint")
        important_indices = np.where(smoothed >= threshold)[0]

        print(f"Window {w}: threshold={threshold}, len(smoothed)={len(smoothed)}, first10={smoothed[:10]}")

        # Extract motifs
        motifs = extract_motifs_with_positions(important_indices, smoothed, sequence, min_len=args.min_len)
        for start, end, peptide, avg_attr in motifs:
            all_motifs.append({
                "Embedding": embedding_name,
                "Model": model_name,
                "Window": w,
                "Start": start,
                "End": end,
                "Sequence": peptide,
                "AvgAttr": avg_attr
            })

    # Save all motifs to CSV
    motifs_df = pd.DataFrame(all_motifs)
    motifs_df.to_csv(args.output, index=False)
    print(f"Saved all motifs to {args.output}")
    print(motifs_df.head())

    # -------------------
    # Plot combined heatmap across all windows for this model
    # -------------------
    if not motifs_df.empty:
        # Initialize coverage matrix
        matrix = pd.DataFrame(0, index=[embedding_name], columns=range(1, args.seq_len+1))

        # Fill coverage from all windows
        for _, row in motifs_df.iterrows():
            start = row["Start"]
            end = row["End"]
            matrix.loc[embedding_name, start:end] += 1

        # Clip for visualization
        matrix = matrix.clip(0, 5)

        # Plot heatmap
        plt.figure(figsize=(15, 1.5))
        sns.heatmap(matrix, cmap="RdPu", cbar_kws={'label': 'Motif coverage'}, linewidths=0)
        plt.xlabel("Residue position")
        plt.ylabel("Embedding")
        plt.title(f"Motif coverage for {embedding_name} (Model={model_name}, All Windows)")
        outpng = f"{embedding_name}_{model_name}_motif_heatmap_all_windows.png"
        plt.tight_layout()
        plt.savefig(outpng, dpi=300)
        plt.close()
        print(f"Saved combined heatmap: {outpng}")

    # Save all motifs to CSV
    motifs_df = pd.DataFrame(all_motifs)
    motifs_df.to_csv(args.output, index=False)
    print(f"Saved all motifs to {args.output}")
    print(motifs_df.head())

if __name__ == "__main__":
    main()
