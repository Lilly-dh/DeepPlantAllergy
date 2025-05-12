# DeepPlantAllergy

**DeepPlantAllergy** is a deep learning framework for allergenicity prediction and motif extraction in plant proteins.  
It leverages transformer-based protein embeddings and interpretability techniques to provide insights into allergenicity at the molecular level.

Allergy is an immune response triggered by specific peptides recognized by immune system effectors.  
To support allergy research and advance the understanding of plant protein allergenicity, we propose **DeepPlantAllergy**, a novel deep learning-based predictor designed to identify allergenic peptides in protein sequences.

DeepPlantAllergy integrates **ESM1b** transformer-based protein embeddings as input and combines:
- **Convolutional Neural Networks (CNNs)** to capture local sequence patterns
- **Bidirectional Long Short-Term Memory (BiLSTM)** networks to model sequential dependencies
- **Multi-head Self-Attention (MHSA)** to enhance predictive performance

Beyond classification, DeepPlantAllergy offers **interpretability** by pinpointing allergenic regions within protein sequences using **Integrated Gradients**, providing valuable insights into the biological mechanisms of allergenicity.



![image](https://github.com/user-attachments/assets/609099b3-9042-477f-a0a0-c7b5ec2a978a)
