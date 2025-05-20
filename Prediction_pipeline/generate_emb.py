import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm  # For the progress bar
import time

# Maximum sequence length for padding
MAX_LENGTH = 1000

# Step 1: Pad the embeddings to the fixed max length
def pad_embeddings(embedding, max_len=MAX_LENGTH):
    seq_len, embedding_dim = embedding.shape
    padded_embedding = np.zeros((max_len, embedding_dim))
    padded_embedding[:seq_len, :] = embedding  # Copy actual values
    return padded_embedding

# Step 2: Define functions for each embedding model
def embed_with_seqvec(sequences):
    from bio_embeddings.embed import SeqVecEmbedder
    embedder = SeqVecEmbedder()
    rawembeddings = [embedder.embed(seq) for seq in tqdm(sequences, desc="Generating SeqVec embeddings")]
    embeddings = [embedding[2] for embedding in rawembeddings]  # Assumes embedding[2] is the last layer
    return embeddings

def embed_with_esm(sequences):
    from bio_embeddings.embed import ESM1bEmbedder
    embedder = ESM1bEmbedder()
    embeddings = [embedder.embed(seq) for seq in tqdm(sequences, desc="Generating ESM embeddings")]
    return embeddings

def embed_with_protbert(sequences):
    from biotransformers import BioTransformers
    bio_trans = BioTransformers(backend="protbert", num_gpus=1)
    embeddings = bio_trans.compute_embeddings(sequences, pool_mode='full')
    return embeddings['full']  # Return the full embeddings from ProtBert

def embed_with_onehot(sequences):
    from bio_embeddings.embed import OneHotEncodingEmbedder
    embedder = OneHotEncodingEmbedder()
    embeddings = [embedder.embed(seq) for seq in tqdm(sequences, desc="Generating OneHot embeddings")]
    return embeddings

# Step 3: Read sequences and IDs from input file (CSV or FASTA)
def read_sequences(file_path):
    seq_ids, sequences, labels = [], [], []
    
    if file_path.endswith(".csv"):
        # Read CSV file
        df = pd.read_csv(file_path)
        required_columns = ['Seq_ID', 'Sequence', 'Label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"❌ Missing required columns in CSV file: {', '.join(missing_columns)}. "
                             "Ensure your CSV has columns 'Seq_ID', 'Sequence', and 'Label'.")
        
        seq_ids = df["Seq_ID"].tolist()
        sequences = df["Sequence"].tolist()
        labels = df["Label"].tolist()  # Assuming 'label' is the name of the column with labels
    
    elif file_path.endswith(".fasta"):
        for record in SeqIO.parse(file_path, "fasta"):
            seq_ids.append(record.id)
            sequences.append(str(record.seq))
        labels = None  # No labels for FASTA files
    
    return seq_ids, sequences, labels

# Step 4: Save embeddings and sequence IDs
def save_embeddings(embeddings, seq_ids, labels, output_prefix):
    embeddings_array = np.stack([pad_embeddings(np.array(embedding)) for embedding in embeddings])
    
    # Adding a timestamp to the output file names
    timestamp = time.strftime("%H%M%S")

    np.save(f"{output_prefix}_{timestamp}_embeddings.npy", embeddings_array)
    np.save(f"{output_prefix}_{timestamp}_sequence_ids.npy", np.array(seq_ids))
    
    if labels is not None:
        np.save(f"{output_prefix}_{timestamp}_labels.npy", np.array(labels))
    
    print(f"✅ Embeddings, sequence IDs, and labels saved as {output_prefix}_{timestamp}_*.npy")


# Step 5: Main function
def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate embeddings for protein sequences using different models.")
    parser.add_argument("input_file", help="Path to input file (CSV or FASTA)")
    parser.add_argument("--model", choices=["SeqVec", "ESM", "ProtBert", "OneHot"], default="SeqVec", help="Embedding model (default: SeqVec)")
    
    args = parser.parse_args()

    # Read sequences from input file
    seq_ids, sequences, labels = read_sequences(args.input_file)
    print(f"📌 Loaded {len(seq_ids)} sequences from {args.input_file}")

    # Generate the output file name based on the input file and model
    file_name = args.input_file.split("/")[-1].split(".")[0]  # Remove path and extension
    output_prefix = f"{file_name}_{args.model}"

    # Generate embeddings based on the selected model
    if args.model == "SeqVec":
        embeddings = embed_with_seqvec(sequences)
    elif args.model == "ESM":
        embeddings = embed_with_esm(sequences)
    elif args.model == "ProtBert":
        embeddings = embed_with_protbert(sequences)
    elif args.model == "OneHot":
        embeddings = embed_with_onehot(sequences)
    
    # Save the embeddings and sequence IDs
    save_embeddings(embeddings, seq_ids, labels, output_prefix)

if __name__ == "__main__":
    main()

