# 🧬 Protein Allergenicity Prediction and Motif Extraction Pipeline

**DeepPlantAllergy** is a deep learning framework for allergenicity prediction and motif extraction in plant proteins.  
It leverages transformer-based protein embeddings and interpretability techniques to provide insights into allergenicity at the molecular level.

Allergy is an immune response triggered by specific peptides recognized by immune system effectors.  
To support allergy research and advance the understanding of plant protein allergenicity, we propose **DeepPlantAllergy**, a novel deep learning-based predictor designed to identify allergenic peptides in protein sequences.

DeepPlantAllergy integrates **ESM1b** transformer-based protein embeddings as input and combines:
- **Convolutional Neural Networks (CNNs)** to capture local sequence patterns
- **Bidirectional Long Short-Term Memory (BiLSTM)** networks to model sequential dependencies
- **Multi-head Self-Attention (MHSA)** to enhance predictive performance

Beyond classification, DeepPlantAllergy offers **interpretability** by pinpointing allergenic regions within protein sequences using **Integrated Gradients**, providing valuable insights into the biological mechanisms of allergenicity.

## ⚙️ 1- Requirements

- **Platform requirement**  
  We trained and tested the model on Linux OS (Ubuntu). Your operating system must be supported by the deep learning framework and related libraries used by the model. For example, our model was implemented using PyTorch 2.4.0+cu121. Please check PyTorch's official OS compatibility to ensure your OS (e.g., Ubuntu, Windows, macOS) is supported.  
  *Note: We have not tested the model on Windows or macOS.*

- **Device requirement**  
  The model was trained on an NVIDIA GeForce RTX 4060 GPU. It is intended to run with GPU acceleration. While it *may* run on CPU if a GPU is not available (`torch.cuda.is_available() == False`), this configuration has not been tested and may result in longer runtimes or unexpected issues.

- **Packages requirement**  
  - Python ≥ 3.8 (tested on Python 3.12)  
  - Conda environments including:  
    - `bio_embeddings`  
    - `bio_transformers`  
  Additional dependencies are listed in the `requirements.txt` file.

## 🖥️ 2- Installation

To get started with this project, you can clone the repository and install its dependencies as follows:


- **Clone the repository**
```bash
git clone https://github.com/Lilly-dh/DeepPlantAllergy.git
```

- **Install required Python packages**
```bash
pip install -r requirements.txt
```

To generate embeddings, we recommend installing `bio_embeddings` and `bio_transformers` in separate Conda environments to avoid potential dependency conflicts. However, the model has also been successfully run in the base environment without environment separation.

For more details on installation and usage, please refer to the official repositories:

    bio_embeddings
    bio_transformers

## 🧠 3 | Training and Testing the Model

### 📦 3.1 | Preparing Embeddings Before Training

Before training the model, you need to generate embeddings from your protein sequences.

#### Input Data Format

Your protein sequences must be provided in a **CSV** file with the following columns:

| Seq_ID | Sequence | Label |
|--------|----------|-------|

- `Seq_ID`: Unique identifier for each protein sequence  
- `Sequence`: The amino acid sequence  
- `Label`: The corresponding binary label (e.g., `1` for allergen, `0` for non-allergen)

#### Generating Embeddings

The module takes the formatted CSV file as input and generates per-sequence embeddings based on the specified model. It outputs one compressed NumPy array file .npz containing the embeddings, the corresponding sequence identifiers and corresponding labels.

```bash
python generate_emb.py --model Model_name sequence.csv
```
Replace Model_name with one of the supported embedding models (OneHot, SeqVec, ProtBert, ESM) and sequence.csv with your formatted input file.

This script will generate the following .npy files:

    filename_Model_name_<timestamp>_data.npz — Sequence embeddings

These output files are required as inputs for the training and testing stages.

### 📚 3.2 | Training

To train the model with your dataset, run the training script with the required arguments, embedding_dim should be provided depending on the embedding model (OneHot → 21, SeqVec → 1024, ProtBert → 1024, ESM → 1280). 

--train_data: Path to the training dataset (.npz file from embedding step).
--val_data: Path to the validation dataset (.npz file).
--batch_size: Number of samples per training batch (default: 16).
--learning_rate: Learning rate for optimization (default: 5e-6).
--weight_decay: Weight decay (L2 regularization) to prevent overfitting (default: 0).
--num_epochs: Number of training epochs (default: 100).
--embedding_dim: Size of the embeddings (depends on embedding model).
--model_type: Model architecture. Options:

- full → CNN + LSTM + Attention
- no_cnn → remove CNN
- no_lstm → remove LSTM
- no_attention → remove Attention

--seed: Random seed for reproducibility (default: 42).

```bash
python train.py --train_data path/to/train_data.npz \
                --val_data path/to/val_data.npz \
                --batch_size 16 \
                --learning_rate 0.000005 \
                --weight_decay 0 \
                --num_epochs 50 \
                --embedding_dim 1024
                --model_type full \
                --seed 42
```
### 📈 3.3| Testing

After training, you can evaluate the model’s performance on a test set using the `test.py` script. Make sure to provide:

- The test data (`.npz`)
- The trained model file (`.pt`)
- The embedding dimension used (e.g., 21 for OneHot)
- A batch size (e.g., 16)
- The model type : full, no_cnn, no_lstm , no_attention
- The seed (42 is default)

#### ✅ Example Command

```bash
python test.py \
  --test_data *_data.npz \
  --model_path best_model.pt \
  --embedding_dim 21 \
  --batch_size 16 \
  --model_type full \
  --seed 42
```
This will output performance metrics such as accuracy, F1 score, MCC, Precisio, Recall, etc and confusion matrix for the test set in a file "model_name_metrics.csv", and a file "model_name_predictions.csv" for per-sequence predictions

## 🚀 4 | Running Predictions


| Step | Description | Script |
|------|-------------|--------|
| 1️⃣ | **Preprocessing** of raw FASTA sequences (removes duplicates, long or invalid entries) | `preprocess.py` |
| 2️⃣ | **Embeddings Generation** using protein language models (OneHot, SeqVec, ESM-1b, ProtBert) | `generate_emb.py` |
| 3️⃣ | **Allergenicity Prediction** using trained models | `predict.py` |
| 4️⃣ | **Residue Attribution Computation** using Integrated Gradients | `compute_attribution.py` |
| 5️⃣ | **Motif Construction** from high-attribution residues | `motif_extract.py` |

---

### 📂 4.1| Preprocessing
It takes a FASTA file as input and performs quality control on the sequences. It removes duplicate sequences and those longer than 1000 amino acid residues. The module outputs a cleaned FASTA file containing the accepted sequences, along with a text file listing the sequence headers that were removed.

Removes:
- Duplicate sequences
- Sequences > 1000 amino acids

**Command:**
```bash
python preprocess.py input_fasta.fasta
```

**Outputs:**
- Cleaned FASTA file
- `.txt` file listing removed sequences headers

---

### 🔬 4.2| Embeddings Generation
It computes sequence embeddings from protein sequences using a selected pretrained protein language model (onehot, seqvec, esm or protbert). It requires the user to activate the relevant Conda environment prior to execution (bio_embeddings for onehot, seqvec and esm, bio_transformers for protbert); the environment and its dependencies must be installed beforehand. The module takes the preprocessed FASTA file as input and generates per-sequence embeddings based on the specified model. It outputs a compressed NumPy array .npz file containing the embeddings and the corresponding sequence identifiers.

Generates embeddings using the selected model:
- `OneHot`
- `SeqVec`
- `ESM`
- `ProtBert`

> ⚠️ Requires activation of the appropriate Conda environment:
> - `bio_embeddings` for OneHot, SeqVec, ESM
> - `bio_transformers` for ProtBert

**Command:**
```bash
python generate_emb.py --model model_name preprocessed_input.fasta
```

**Outputs:**
- `filename_model_timestamp_data_.npz`

---

### 🔍 4.3| Allergenicity Prediction
It uses one of the trained models to classify protein sequences based on their previously generated embeddings. It takes as input the .npz data file previosuly generated. The user specifies the embedding model used during embedding generation to ensure compatibility with the correct trained model. 
The user can choose from 16 available model configurations (full, no_cnn, no_lstm and no_attention) and embeddings (Onehot, Seqvec, ESM and Protbert). 
The module outputs a CSV file listing the sequence identifiers, their corresponding probability (predicted probability of allergenicity), prediction (predicted class label), and comment where probability higher than 0.8 is considered "High probability allergen", probability between 0.5 and 0.8, "potentially allergen" , and is labeled "probably not allergen" when probability is lower than 0.5.

**Command:**
```bash
python predict.py \
    --embedding seqvec \
    --model_type full \
    --input *.npz
    --batch_size 16 \
    --seed 42 
```

**Outputs (CSV):**
- Sequence ID
- Probability of allergenicity
- Predicted label
- Comment (e.g., "High probability allergen", etc.)

---

### 🧠 4.4| Residue Attribution
It uses the prediction model and Integrated Gradient to compute attributions to each residue, it returns the sum of raw attribution across embedding dimensions, as well as smoothed attributions and normalized attributions. It outputs a CSV file per sequence listing the residues and their corresponding attributions. The code uses automatically window-size=11 for smoothing.

**Command:**
```bash
python compute_attribution.py \
  --fasta_file preprocessed_input.fasta \
  --prediction_file predictions.csv \
  --embedding_file embeddings.npy \
  --output_dir attribution_output/ \
  --model_name model_name
```

**Outputs:**
- One CSV per sequence with raw, smoothed, and normalized attribution scores

---

### 🧬 4.5| Motif Construction
Processes raw attribution scores to identify potential motifs. The user can define parameters such as max_gap (default = 1) and max_motif_length (default = 20). The output consists of two CSV files: one with raw signal data and another with merged motifs, including motif start and end positions, length, and gap-related statistics such as gap_num and gap_density.

**Command:**
```bash
python motif_extract.py \
  --attribution_file attributions.csv \
```

**Outputs:**
- Raw signal file
- Merged motif file with positions, lengths, gaps

---

## 🧾 Citation

If you use this pipeline in your research, please cite our upcoming manuscript submitted to *Journal Name*.
