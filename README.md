# ECG-MIT-BIH-Arrhythmia-multi-beat-classification-using-GNN
CardioECGNet-GNN is a graph-based deep learning framework for automated ECG arrhythmia classification.  The method models ECG beats as nodes in a graph and uses Graph Neural Networks (GCNs) to capture  inter-beat relationships, improving robustness and interpretability.
# CardioECGNet-GNN


## Overview
CardioECGNet-GNN is a graph-based deep learning framework for automated ECG arrhythmia classification. 
The method models ECG beats as nodes in a graph and uses Graph neural Networks (GNNs) to capture 
inter-beat relationships, improving robustness and interpretability.

## Key Contributions
- ECG beats represented as graph nodes
- k-NN based graph construction
- Three-layer GNN architecture
- Robust handling of class imbalance
- Evaluated on MIT-BIH Arrhythmia Dataset

##  Dataset
- MIT-BIH Arrhythmia Database
- ~109,000 annotated heartbeats
- Balanced dataset used for training

> Dataset is not included.

## Methodology
1. ECG preprocessing and normalization
2. Feature imputation using KNN
3. k-NN graph construction (k = 5)
4. Graph Convolutional Network training
5. Arrhythmia classification

##  Model Architecture
- Input features: 360
- Hidden layers: 3 GCNConv layers
- Hidden channels: 246
- Dropout: 0.5
- Output classes: 5

##  Experimental Setup
- Train/Test split: 70/30
- Optimizer: Adam
- Learning rate: 0.01
- Loss function: NLLLoss
- Epochs: 500

## Results
The GNN-based model demonstrates effective ECG arrhythmia classification by leveraging relational learning 
among ECG beats.

