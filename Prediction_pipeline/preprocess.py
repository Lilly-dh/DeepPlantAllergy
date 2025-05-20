import os
import argparse
from Bio import SeqIO
from datetime import datetime

def read_fasta(fasta_file):
    """Read sequences from a FASTA file and return a list of (ID, sequence) tuples."""
    return [(record.id, str(record.seq)) for record in SeqIO.parse(fasta_file, "fasta")]

def find_and_remove_non_standard_sequences(sequences):
    """Identify and remove sequences containing non-standard amino acids."""
    standard_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    clean_sequences = []
    removed_non_standard = []

    for seq_id, seq in sequences:
        if any(aa.upper() not in standard_amino_acids for aa in seq):
            removed_non_standard.append(seq_id)
        else:
            clean_sequences.append((seq_id, seq))
    
    return clean_sequences, removed_non_standard

def remove_duplicates(sequences):
    """Remove duplicate sequences based on sequence IDs."""
    seen = set()
    unique_sequences = []
    removed_duplicates = []
    
    for seq_id, seq in sequences:
        if seq_id in seen:
            removed_duplicates.append(seq_id)  # Track removed duplicate IDs
        else:
            unique_sequences.append((seq_id, seq))
            seen.add(seq_id)

    return unique_sequences, removed_duplicates

def filter_by_length(sequences, max_length=1000):
    """Filter sequences longer than max_length amino acids."""
    filtered_sequences = [(seq_id, seq) for seq_id, seq in sequences if len(seq) <= max_length]
    removed_long = [seq_id for seq_id, seq in sequences if len(seq) > max_length]
    return filtered_sequences, removed_long

def save_to_fasta(sequences, output_file):
    """Save sequences to a FASTA file."""
    with open(output_file, 'w') as f:
        for seq_id, seq in sequences:
            f.write(f">{seq_id}\n{seq}\n")

def save_removed_headers(removed_ids, output_file):
    """Save removed sequence IDs to a text file."""
    with open(output_file, 'w') as f:
        for seq_id in removed_ids:
            f.write(f"{seq_id}\n")

def generate_timestamp():
    """Generate a timestamp for file naming."""
    return datetime.now().strftime("%H%M%S")

def main():
    parser = argparse.ArgumentParser(description="Preprocess a FASTA file by removing duplicates, filtering by length, and removing sequences with non-standard amino acids.")
    parser.add_argument("input_fasta", help="Path to the input FASTA file")
    parser.add_argument("--max_length", type=int, default=1000, help="Maximum allowed sequence length (default: 1000)")
    
    args = parser.parse_args()

    # Generate timestamp for unique filenames
    timestamp = generate_timestamp()
    
    # Derive output file names based on the input FASTA file name with timestamp
    input_file_name = os.path.basename(args.input_fasta)
    input_name = os.path.splitext(input_file_name)[0]  # Extract file name without extension
    
    output_file_name = f"{input_name}_processed_{timestamp}.fasta"
    removed_headers_file_name = f"{input_name}_removed_headers_{timestamp}.txt"
    
    print(f"Reading sequences from {args.input_fasta}...")
    sequences_list = read_fasta(args.input_fasta)
    total_sequences = len(sequences_list)
    print(f"Total sequences before processing: {total_sequences}")

    # Remove sequences with non-standard amino acids
    cleaned_sequences, removed_non_standard = find_and_remove_non_standard_sequences(sequences_list)
    num_non_standard = len(removed_non_standard)
    print(f"Sequences with non-standard amino acids removed: {num_non_standard}")

    # Remove duplicates
    unique_sequences, removed_duplicates = remove_duplicates(cleaned_sequences)
    num_duplicates = len(removed_duplicates)
    print(f"Duplicate sequences removed: {num_duplicates}")

    # Filter by length
    filtered_sequences, removed_long_ids = filter_by_length(unique_sequences, args.max_length)
    num_long = len(removed_long_ids)
    print(f"Sequences longer than {args.max_length} removed: {num_long}")

    # Save the cleaned sequences
    save_to_fasta(filtered_sequences, output_file_name)
    print(f"Preprocessed sequences saved to {output_file_name}")

    # Save removed headers to a dynamically named text file
    save_removed_headers(removed_duplicates + removed_long_ids + removed_non_standard, removed_headers_file_name)
    print(f"List of removed sequence headers saved to {removed_headers_file_name}")

    # Print final statistics
    print("\n--- Processing Summary ---")
    print(f"Total sequences before processing: {total_sequences}")
    print(f"Duplicate sequences removed: {num_duplicates}")
    print(f"Sequences with non-standard amino acids removed: {num_non_standard}")
    print(f"Sequences longer than {args.max_length} removed: {num_long}")
    print(f"Total sequences after processing: {len(filtered_sequences)}")

if __name__ == "__main__":
    main()

