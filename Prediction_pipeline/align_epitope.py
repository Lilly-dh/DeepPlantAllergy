#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime
from Bio import pairwise2

def align_motif_to_epitope(motif_seq, epitope_seq):
    epitope_seq = epitope_seq.upper()
    alignments = pairwise2.align.globalms(motif_seq, epitope_seq, 2, -1, -2, -0.5)
    best = alignments[0]
    aligned_motif, aligned_epitope, score, start, end = best

    aligned_matches = sum(1 for a, b in zip(aligned_motif, aligned_epitope) if a != '-' and b != '-')
    identity = sum(1 for a, b in zip(aligned_motif, aligned_epitope) if a == b and a != '-') / aligned_matches if aligned_matches > 0 else 0.0
    epitope_coverage = aligned_matches / len(epitope_seq)
    ratio = len(motif_seq) / len(epitope_seq)

    return {
        "motif_seq": motif_seq,
        "epitope_seq": epitope_seq,
        "aligned_motif": aligned_motif,
        "aligned_epitope": aligned_epitope,
        "score": score,
        "identity": round(identity, 3),
        "epitope_coverage": round(epitope_coverage, 3),
        "motif_to_epitope_ratio": round(ratio, 3)
    }

def should_align_by_position(motif_start, motif_end, epitope_start, epitope_end, max_distance=20):
    return (
        abs(motif_start - epitope_start) <= max_distance or
        abs(motif_end - epitope_end) <= max_distance
    )

def has_common_kmer(motif, epitope, k=3):
    motif = motif.replace("-", "")
    epitope = epitope.upper()
    motif_kmers = {motif[i:i+k] for i in range(len(motif) - k + 1)}
    return any(kmer in epitope for kmer in motif_kmers)

def selective_motif_alignment(motif_df, iedb_df, max_distance=20, k=3):
    results = []
    for _, motif_row in motif_df.iterrows():
        for _, epitope_row in iedb_df.iterrows():
            if should_align_by_position(
                motif_row["Start"], motif_row["End"],
                epitope_row["epitope_start"], epitope_row["epitope_end"],
                max_distance=max_distance
            ) or has_common_kmer(motif_row["Motif"], epitope_row["epitope_seq"], k=k):
                
                result = align_motif_to_epitope(motif_row["Motif"], epitope_row["epitope_seq"])
                result.update({
                    "motif_start": motif_row["Start"],
                    "motif_end": motif_row["End"],
                    "motif_length": motif_row["Motif_Length"],
                    "num_gaps": motif_row["Num_Gaps"],
                    "gap_density": motif_row["Gap_Density"],
                    "epitope_start": epitope_row["epitope_start"],
                    "epitope_end": epitope_row["epitope_end"],
                    "epitope_id": epitope_row["epitope_id"],
                    "num_assays": epitope_row["num_assays"],
                    "response_freq": epitope_row["response_freq"]
                })
                results.append(result)
    return pd.DataFrame(results)

def auto_output_name(input_file, prefix):
    basename = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{basename}_{timestamp}.csv"

def clean_iedb_epitope_file(iedb_path):
    iedb_df = pd.read_csv(iedb_path)
    cols = ["Epitope ID", "Sequence", "Mapped Start Position", "Mapped End Position", "Subjects Tested", "Response Freq."]
    iedb_df = iedb_df[cols]
    iedb_df.columns = ["epitope_id", "epitope_seq", "epitope_start", "epitope_end", "num_assays", "response_freq"]
    iedb_df = iedb_df[~iedb_df["epitope_seq"].str.contains(r'\d')]  # Remove conformational epitopes
    return iedb_df

def main():
    parser = argparse.ArgumentParser(description="Align predicted motifs to IEDB epitopes.")
    parser.add_argument("--motif_csv", required=True, help="CSV file with extracted motifs")
    parser.add_argument("--iedb_csv", required=True, help="IEDB epitope export CSV file")
    parser.add_argument("--max_distance", type=int, default=20, help="Max positional distance to allow alignment (default=20)")
    parser.add_argument("--kmer", type=int, default=3, help="Minimum shared k-mer length for alignment (default=3)")
    args = parser.parse_args()

    motif_df = pd.read_csv(args.motif_csv)
    iedb_df = clean_iedb_epitope_file(args.iedb_csv)

    alignment_df = selective_motif_alignment(motif_df, iedb_df, max_distance=args.max_distance, k=args.kmer)

    all_output = auto_output_name(args.motif_csv, "motif_epitope_alignment")
    best_output = auto_output_name(args.motif_csv, "best_motif_epitope_alignment")

    alignment_df.to_csv(all_output, index=False)
    print(f"[INFO] Saved all alignments to: {all_output}")

    best_per_motif = alignment_df.sort_values(by="score", ascending=False).groupby("motif_seq").head(1)
    best_per_motif.to_csv(best_output, index=False)
    print(f"[INFO] Saved best alignment per motif to: {best_output}")

if __name__ == "__main__":
    main()
