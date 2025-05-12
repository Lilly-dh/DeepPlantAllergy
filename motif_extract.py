#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime

def extract_motifs_with_positions(important_indices):
    """Extract contiguous motifs and their positions from important residue indices."""
    motifs = []
    if len(important_indices) == 0:
        return motifs

    start = important_indices[0]

    for i in range(1, len(important_indices)):
        if important_indices[i] != important_indices[i - 1] + 1:
            motifs.append((start, important_indices[i - 1]))
            start = important_indices[i]

    motifs.append((start, important_indices[-1]))
    return motifs

def merge_and_combine_motifs(motif_df, max_gap=1, max_motif_length=30):
    """Merge adjacent motifs, allowing small gaps and a max motif length."""
    merged = []
    current_motif = ""
    current_start = None
    current_end = None
    prev_end = None

    for idx, row in motif_df.iterrows():
        start, end, motif = row["Start"], row["End"], row["Motif"]

        if current_motif == "":
            current_motif = motif
            current_start = start
            current_end = end
            prev_end = end
        else:
            gap = start - prev_end - 1
            if 0 <= gap <= max_gap and (end - current_start + 1 + gap) <= max_motif_length:
                current_motif += "-" * gap + motif
                current_end = end
                prev_end = end
            else:
                motif_length = len(current_motif)
                num_gaps = current_motif.count("-")
                gap_density = round(num_gaps / motif_length, 2) if motif_length > 0 else 0.0

                merged.append({
                    "Start": current_start,
                    "End": current_end,
                    "Motif": current_motif,
                    "Motif_Length": motif_length,
                    "Num_Gaps": num_gaps,
                    "Gap_Density": gap_density
                })

                current_motif = motif
                current_start = start
                current_end = end
                prev_end = end

    if current_motif:
        motif_length = len(current_motif)
        num_gaps = current_motif.count("-")
        gap_density = round(num_gaps / motif_length, 2) if motif_length > 0 else 0.0

        merged.append({
            "Start": current_start,
            "End": current_end,
            "Motif": current_motif,
            "Motif_Length": motif_length,
            "Num_Gaps": num_gaps,
            "Gap_Density": gap_density
        })

    return pd.DataFrame(merged)

def auto_name_output(input_file, suffix="_motifs"):
    """Create output filename with timestamp, based on input filename."""
    basename = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{basename}{suffix}_{timestamp}.csv"

def main():
    parser = argparse.ArgumentParser(description="Extract motifs from raw attribution scores.")
    parser.add_argument("--attribution_file", required=True, help="Input CSV file with attribution scores")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold above which residues are considered important (default: 0.0)")
    parser.add_argument("--max_gap", type=int, default=1, help="Maximum gap allowed when merging motifs (default: 1)")
    parser.add_argument("--max_motif_length", type=int, default=20, help="Maximum allowed length of a merged motif (default: 20)")
    
    args = parser.parse_args()

    df = pd.read_csv(args.attribution_file)
    attribution_column = "Raw Attribution"

    if attribution_column not in df.columns:
        raise ValueError(f"Required column '{attribution_column}' not found in the CSV file.")

    important_indices = np.where(df[attribution_column] > args.threshold)[0]

    motifs_with_positions = extract_motifs_with_positions(important_indices)

    motif_info = [{
        'Motif': "".join(df.iloc[start:end + 1]["Residue"]),
        'Start': start + 1,
        'End': end + 1,
        'Length': (end - start) + 1
    } for start, end in motifs_with_positions]

    results_df = pd.DataFrame(motif_info)

    combined_df = merge_and_combine_motifs(results_df, max_gap=args.max_gap, max_motif_length=args.max_motif_length)

    output_file = auto_name_output(args.attribution_file)

    combined_df.to_csv(output_file, index=False)

    print(f"[INFO] Motif extraction complete. Output saved to: {output_file}")
    print(combined_df)

if __name__ == "__main__":
    main()
