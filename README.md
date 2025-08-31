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
  We trained and tested the model on Linux (Ubuntu). Your operating system must be supported by the deep learning framework and related libraries used by the model. Our implementation uses PyTorch 2.4.0 with CUDA 12.1. Please check PyTorch’s official compatibility page to ensure that both your OS (e.g., Ubuntu, Windows, macOS) and your CUDA version are supported.

⚠️ Important: The model requires a CUDA-enabled GPU for training and inference. We have not tested the model on Windows or macOS.

- **Device requirement**  
  We trained and tested the model on Linux (Ubuntu) with an NVIDIA GeForce RTX 4060 GPU. The model is intended to run only with GPU acceleration. A CUDA-enabled GPU and compatible CUDA installation are required (torch.cuda.is_available() must return True).
⚠️ Note: Running on CPU-only is not supported and will cause the model to fail. We have not tested the model on Windows or macOS.

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
python generate_embeddings.py --model Model_name filename.csv
```
Replace Model_name with one of the supported embedding models (OneHot, SeqVec, ProtBert, ESM) and sequence.csv with your formatted input file.

This script will generate the following .npz file:

    filename_Model_name_<timestamp>_data.npz

These output files are required as inputs for the training and testing stages.

### 📚 3.2 | Training

To train the model with your own dataset, run the training script and specify the necessary arguments. The most important one is --embedding_dim, which must match the embedding model you used during preprocessing (OneHot = 21, SeqVec = 1024, ProtBert = 1024, ESM = 1280).

You will also need to provide the paths to your training and validation datasets (both in .npz format, generated during the embedding step). Training parameters such as batch size (default: 16), learning rate (default: 5e-6), weight decay for regularization (default: 0), and the number of epochs (default: 100) can be customized through their respective arguments.

The script also allows you to choose the model architecture via the --model_type flag. Options include the full model (CNN + LSTM + Attention), or ablation variants where one of these components is removed (no_cnn, no_lstm, no_attention). Finally, you can set a random seed (default: 42) to ensure reproducibility of results.

```bash
python training.py --train_data path/to/train_data.npz \
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

After training, you can evaluate the model on a held-out test set using the test.py script. To do this, provide the path to your test dataset (in .npz format), the trained model file (saved as .pt), and the embedding dimension that matches your embedding model (e.g., 21 for OneHot, 1024 for SeqVec or ProtBert, 1280 for ESM).

You can also adjust the batch size (default: 16) and specify which architecture to evaluate (full, no_cnn, no_lstm, or no_attention). For reproducibility, the script accepts a random seed argument (default: 42).

#### ✅ Example Command

```bash
python testing.py --test_data *_data.npz  --model_path best_model.pt  --embedding_dim 21  --batch_size 16  --model_type full  --seed 42
```
This will output performance metrics such as accuracy, F1 score, MCC, Precisio, Recall, etc and confusion matrix for the test set in a file "model_name_metrics.csv", and a file "model_name_predictions.csv" for per-sequence predictions

## 🚀 4 | Running Predictions


| Step | Description | Script |
|------|-------------|--------|
| 1️⃣ | **Preprocessing** of raw FASTA sequences (removes duplicates, long or invalid entries) | `preprocess.py` |
| 2️⃣ | **Embeddings Generation** using protein language models (OneHot, SeqVec, ESM-1b, ProtBert) | `generate_embeddings.py` |
| 3️⃣ | **Allergenicity Prediction** using trained models | `prediction.py` |
| 4️⃣ | **Residue Attribution Computation** using Integrated Gradients | `residue_attribution.py` |
| 5️⃣ | **Motif Construction** from high-attribution residues | `motif_extractor.py` |
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
It computes sequence embeddings from protein sequences using a selected pretrained protein language model (onehot, seqvec, esm or protbert). It requires the user to activate the relevant Conda environment prior to execution (bio_embeddings for onehot, seqvec and esm, bio_transformers for protbert); the environment and its dependencies must be installed beforehand. The module takes the preprocessed FASTA file as input and generates per-sequence embeddings based on the specified model. It outputs a compressed NumPy array (.npz) file containing the embeddings and the corresponding sequence identifiers.

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
python generate_embeddings.py --model model_name preprocessed_input.fasta
```

**Outputs:**
- `filename_model_timestamp_data.npz`

---

### 🔍 4.3| Allergenicity Prediction
It uses one of the trained models to classify protein sequences based on their previously generated embeddings. It takes as input the .npz data file previosuly generated. The user specifies the embedding model used during embedding generation to ensure compatibility with the correct trained model. 
The user can choose from 16 available model configurations (full, no_cnn, no_lstm and no_attention) and embeddings (Onehot, Seqvec, ESM and Protbert). 
The module outputs a CSV file listing the sequence identifiers, their corresponding probability (predicted probability of allergenicity), prediction (predicted class label), and comment where probability higher than 0.8 is considered "High probability allergen", probability between 0.5 and 0.8, "potentially allergen" , and is labeled "probably not allergen" when probability is lower than 0.5.

**Command:**
```bash
python prediction.py \
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
It uses the prediction model and Integrated Gradient to compute attributions to each residue, it returns the sum of raw attribution across embedding dimensions, as well as smoothed attributions and normalized attributions. It outputs one CSV per sequence listing each residue along with the sum and mean of attributions across embedding dimensions.

**Command:**
```bash
python residue_attribution.py --fasta_file  --embedding_file filename.npz --output_dir --embedding_type [OneHot, SeqVec, ProtBert, or ESM] \
    --model_type [full, no_cnn, no_lstm, or no_attention] --seed (default: 42)
```

**Outputs:**
- One CSV per sequence with aggregated and averaged attribution scores

---

### 🧬 4.5| Motif Construction
It processes raw per-residue attribution scores to identify potential motifs. Attribution scores from a CSV are smoothed using a configurable window, and residues above a chosen threshold are selected to define contiguous segments meeting a minimum length. For each motif, the script computes the average attribution and saves all results—including embedding, model type, window size, positions, sequence, and average attribution—to a CSV. It also generates a heatmap PNG showing motif coverage across the sequence for the specified embedding and model.

**Command:**
```bash
python motif_extractor.py  --csv attribution_example.csv   --windows 5 7 9 11  --threshold percentile --percentile 80  --seq-len 150  --min-len 3    --output motifs_output.csv

```

**Outputs:**
- CSV of motifs extracted with their start and end position, average attribution score and Model name, embedding and window size.
- Heatmap of motif coverage

---

## 🧾 Citation

If you use this pipeline in your research, please cite our upcoming manuscript submitted to *Journal Name*.
