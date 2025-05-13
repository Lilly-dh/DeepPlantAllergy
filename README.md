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

<div align="center">
  <img src="Pipeline_new.pdf" alt="Pipeline Overview" width="600"/>
</div>

## 📌 Overview of Pipeline Steps

| Step | Description | Script |
|------|-------------|--------|
| 1️⃣ | **Preprocessing** of raw FASTA sequences (removes duplicates, long or invalid entries) | `preprocess.py` |
| 2️⃣ | **Embeddings Generation** using protein language models (OneHot, SeqVec, ESM-1b, ProtBert) | `generate_emb.py` |
| 3️⃣ | **Allergenicity Prediction** using trained models | `predict.py` |
| 4️⃣ | **Residue Attribution Computation** using Integrated Gradients | `compute_attribution.py` |
| 5️⃣ | **Motif Construction** from high-attribution residues | `motif_extract.py` |
| 6️⃣ | **Epitope Alignment** against IEDB validated epitopes | `align_epitope.py` |

---

## 📂 1. Preprocessing

Removes:
- Duplicate sequences
- Sequences > 1000 amino acids
- Sequences with non-standard amino acids

**Command:**
```bash
python preprocess.py input_fasta.fasta
```

**Outputs:**
- Cleaned FASTA file
- `.txt` file listing removed sequences

---

## 🔬 2. Embeddings Generation

Generates embeddings using the selected model:
- `onehot`
- `seqvec`
- `esm`
- `protbert`

> ⚠️ Requires activation of the appropriate Conda environment:
> - `bio_embeddings` for OneHot, SeqVec, ESM
> - `bio_transformers` for ProtBert

**Command:**
```bash
python generate_emb.py --model model_name preprocessed_input.fasta
```

**Outputs:**
- `embeddings.npy`
- `sequence_ids.npy`

---

## 🔍 3. Allergenicity Prediction

Uses a trained model to classify sequences based on embeddings.

**Command:**
```bash
python predict.py --model model_name --input_emb embeddings.npy --input_ids sequence_ids.npy
```

**Outputs (CSV):**
- Sequence ID
- Probability of allergenicity
- Predicted label
- Comment (e.g., "High probability allergen", etc.)

---

## 🧠 4. Residue Attribution

Applies Integrated Gradients to identify residue-level contributions.

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

## 🧬 5. Motif Construction

Extracts potential allergenic motifs from high-attribution regions.

**Command:**
```bash
python motif_extract.py \
  --attribution_file attributions.csv \
  --threshold 0 \
  --max_gap 1 \
  --max_motif_length 20
```

**Outputs:**
- Raw signal file
- Merged motif file with positions, lengths, gaps

---

## 🔗 6. Epitope Alignment

Aligns predicted motifs to validated epitopes from IEDB.

**Command:**
```bash
python align_epitope.py \
  --motif_csv motifs.csv \
  --iedb_csv iedb_export.csv
```

**Output:**
- Best-matching epitope–motif pair with alignment score

---

## ⚙️ Requirements

- Python ≥ 3.8
- Conda environments:
  - `bio_embeddings`
  - `bio_transformers`
- Dependencies listed in environment YAML files (not provided here but should be added to repo)

---

## 🧾 Citation

If you use this pipeline in your research, please cite our upcoming manuscript submitted to *Nucleic Acids Research*.
