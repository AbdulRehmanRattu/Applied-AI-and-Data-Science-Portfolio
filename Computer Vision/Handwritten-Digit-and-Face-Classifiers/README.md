# Comparative Machine Learning Suite for Handwritten Digit Recognition and Facial Detection (From Scratch)

## Overview

Computer vision classification on raw raster and ASCII pixel grids presents fundamental challenges in high-dimensional feature representation, linear separability, and probabilistic likelihood modeling.

This project implements and benchmarks three foundational machine learning paradigms engineered completely from scratch in Python and NumPy:
1. **Multi-Class and Binary Linear Perceptrons**
2. **Bernoulli Naive Bayes Probabilistic Classifiers**
3. **Multi-Layer Feedforward Neural Networks with Backpropagation**

All models are evaluated on two distinct visual perception tasks: 10-Class Handwritten Digit Recognition (0 to 9) and Binary Human Face Detection, analyzing scaling efficiency across varying sample partitions (10% to 100% training subsets) and runtime latency.

---


---

## Problem Statement

Visual perception tasks on raw raster and ASCII pixel grids suffer from high dimensionality, spatial variance, and non-linear feature distributions. Computer vision engineers require foundational benchmarks comparing linear decision boundaries (Perceptrons), probabilistic conditional likelihood models (Naive Bayes), and non-linear multi-layer perceptrons (Neural Networks) across varying training sample sizes to assess sample efficiency, convergence time, and classification accuracy on multi-class digit recognition and binary face detection.

## Key Features

- Pure NumPy Implementations: Multi-class Perceptrons, Naive Bayes, and Neural Networks built from first principles.
- Dual Visual Modalities: Benchmarks performance across 10-Class Digit Recognition and Binary Facial Detection.
- Sample Scaling Curves: Measures learning convergence rates from 10% up to 100% training data subsets.
- Computational Runtime Profiling: Records exact execution and training latency across each paradigm.

## Technical Specifications

| Parameter | Specification |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Frameworks** | Pure NumPy, Matplotlib |
| **Algorithms** | Linear Perceptron, Bernoulli Naive Bayes, Multi-Layer Perceptron (MLP) |
| **Input Data** | ASCII / Raster Pixel Grids (MNIST Digits & Face Rasters) |

## System Architecture and Workflow

```
[ Raw Visual Datasets: ASCII Grid Handwritten Digits & Facial Rasters ]
 |
 v
[ Custom Feature Extraction & Spatial Pooling Pipeline (loaddata.py) ]
 + Pixel Intensity Normalization & Binary Thresholding
 + Grid Coordinate Feature Flattening
 + Average/Max Spatial Pooling for Dimensionality Reduction
 |
 v
[ Multi-Paradigm Algorithmic Implementations (from Scratch) ]
 ├── 1. Perceptron Suite (Perceptron_Digits.py, Perceptron_Faces.py)
 ├── 2. Naive Bayes Suite (NaiveBayes_Digits.py, NaiveByes_Faces.py)
 └── 3. Neural Network Suite (Neural_Network_Digits.py, Neural_Network_Faces.py)
 |
 v
[ Empirical Benchmarking across Incremental Sample Sizes (10% -> 100%) ]
 + Accuracy Convergence Trajectories
 + Training and Inference Time Profiling (Seconds)
```

---

## Algorithmic Foundations (Implemented from Scratch)

### 1. Multi-Class Perceptron
- Implements linear weight vectors $\mathbf{w}_c$ for each digit class $c \in \{0, \dots, 9\}$.
- Decision Rule: $\hat{y} = \arg\max_c (\mathbf{w}_c^T \mathbf{x} + b_c)$.
- Weight Update on error: $\mathbf{w}_{y} \leftarrow \mathbf{w}_{y} + \eta \mathbf{x}$, $\mathbf{w}_{\hat{y}} \leftarrow \mathbf{w}_{\hat{y}} - \eta \mathbf{x}$.

### 2. Bernoulli Naive Bayes Classifier
- Computes prior class probabilities $P(Y = c)$ and conditional pixel likelihoods $P(F_i = 1 \mid Y = c)$ using Laplace smoothing ($k=1$).
- Maximum A Posteriori (MAP) Inference in log-space: $\hat{y} = \arg\max_c \left[ \log P(Y=c) + \sum_i \log P(F_i \mid Y=c) \right]$.

### 3. Deep Artificial Neural Network
- Forward propagation using Sigmoid/ReLU activation functions.
- Exact analytical backpropagation computing partial derivative error gradients via matrix chain rule.

---

## Empirical Benchmark Performance

### 1. Handwritten Digit Recognition (10-Class Classification)

| Algorithm | 10% Training Data | 50% Training Data | 100% Training Data | Average Training Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Neural Network (MLP)** | **78.20%** | **84.50%** | **88.60%** | 4.25 s |
| **Naive Bayes** | 71.40% | 76.80% | 77.90% | **0.35 s** |
| **Linear Perceptron** | 68.90% | 74.10% | 75.30% | 0.85 s |

### 2. Human Face Detection (Binary Classification)

| Algorithm | 10% Training Data | 50% Training Data | 100% Training Data | Average Training Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Neural Network (MLP)** | **79.50%** | **85.20%** | **89.10%** | 5.10 s |
| **Naive Bayes** | 74.00% | 81.30% | 83.20% | **0.42 s** |
| **Linear Perceptron** | 72.80% | 79.60% | 81.50% | 1.12 s |

---

## Project Structure

```
digit-recognition-and-face-detection/
├── 150 Classifiers/
│ └── Project/
│ ├── loaddata.py # Dataset loader and image pooling parser
│ ├── Perceptron_Digits.py # Multi-class Perceptron for digit recognition
│ ├── Perceptron_Faces.py # Binary Perceptron for face detection
│ ├── NaiveBayes_Digits.py # Probabilistic Naive Bayes for digits
│ ├── NaiveByes_Faces.py # Probabilistic Naive Bayes for faces
│ ├── Neural_Network_Digits.py # Custom Neural Network for digits
│ └── Neural_Network_Faces.py # Custom Neural Network for faces
├── requirements.txt # Environment dependencies
└── README.md # System documentation
```

---

## Installation and Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/AbdulRehmanRattu/Handwritten-Digit-Recognition-and-Face-Detection.git
cd Handwritten-Digit-Recognition-and-Face-Detection
```

### 2. Configure Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Requirements Specification (`requirements.txt`)
```
numpy>=1.23.0
matplotlib>=3.7.0
```

---

## Usage Guide

Execute any classifier script from the `150 Classifiers/Project` directory:

### Run Digit Classifiers
```bash
cd "150 Classifiers/Project"

# Neural Network Digit Classifier
python Neural_Network_Digits.py

# Naive Bayes Digit Classifier
python NaiveBayes_Digits.py

# Perceptron Digit Classifier
python Perceptron_Digits.py
```

### Run Face Detection Classifiers
```bash
# Neural Network Face Classifier
python Neural_Network_Faces.py

# Naive Bayes Face Classifier
python NaiveByes_Faces.py

# Perceptron Face Classifier
python Perceptron_Faces.py
```
