# Custom Deep Neural Network Classifier for Breast Cancer Diagnostics (NumPy from Scratch)

## Overview

While high-level frameworks (e.g. PyTorch, TensorFlow) abstract gradient computation, implementing deep neural networks from foundational mathematical first principles provides complete transparency, precise numerical stability control, and minimal runtime overhead.

This project implements a multi-layer deep feedforward artificial neural network (MLP) engineered entirely from scratch using pure NumPy. The implementation features explicit analytical backpropagation, matrix-calculus gradient derivations, Sigmoid activation derivatives, and vector-accelerated batch gradient descent to diagnose malignant versus benign tumors on the Wisconsin Diagnostic Breast Cancer dataset, achieving a verified test classification accuracy of 90.35%.

---


---

## Problem Statement

Early and accurate differentiation between benign and malignant breast tumors is vital for oncology treatment planning and patient survival. While modern deep learning relies on black-box frameworks, clinical diagnostic systems demand mathematically transparent, highly regularized neural network classifiers derived directly from first-principles matrix calculus to process fine needle aspirate (FNA) cytology measurements with high diagnostic reliability.

## Key Features

- Pure NumPy from First Principles: Implements deep forward and backward propagation without autograd libraries.
- Analytical Matrix Calculus Gradients: Explicit chain rule derivative formulas for all hidden and output layers.
- Verified 90.35% Holdout Accuracy: Benchmarked on standardized Wisconsin Breast Cancer biometric features.
- Vectorized Batch Gradient Descent: Matrix-accelerated numerical training achieving convergence in 1,000 epochs.

## System Architecture and Workflow & Mathematical Derivation

```
[ Input Layer: 30 Standardized Biometric Features ]
 |
 v (W1: 30x128, b1: 1x128)
[ Hidden Layer 1: 128 Neurons + Sigmoid Activation ]
 |
 v (W2: 128x128, b2: 1x128)
[ Hidden Layer 2: 128 Neurons + Sigmoid Activation ]
 |
 v (W3: 128x1, b3: 1x1)
[ Output Layer: 1 Neuron + Sigmoid Binary Probability ]
 |
 v
[ Analytical Backpropagation & Parameter Update via Gradient Descent ]
```

### Mathematical Equations

#### 1. Forward Propagation
$$\mathbf{a}_1 = \mathbf{X}\mathbf{W}_1 + \mathbf{b}_1, \quad \mathbf{h}_1 = \sigma(\mathbf{a}_1)$$
$$\mathbf{a}_2 = \mathbf{h}_1\mathbf{W}_2 + \mathbf{b}_2, \quad \mathbf{h}_2 = \sigma(\mathbf{a}_2)$$
$$\mathbf{a}_3 = \mathbf{h}_2\mathbf{W}_3 + \mathbf{b}_3, \quad \hat{\mathbf{y}} = \sigma(\mathbf{a}_3)$$

#### 2. Backward Propagation (Chain Rule Derivations)
$$\boldsymbol{\delta}_3 = (\hat{\mathbf{y}} - \mathbf{y}) \odot \hat{\mathbf{y}} \odot (1 - \hat{\mathbf{y}})$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_3} = \frac{1}{N}\mathbf{h}_2^T \boldsymbol{\delta}_3, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}_3} = \frac{1}{N}\sum \boldsymbol{\delta}_3$$
$$\boldsymbol{\delta}_2 = (\boldsymbol{\delta}_3 \mathbf{W}_3^T) \odot \mathbf{h}_2 \odot (1 - \mathbf{h}_2)$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \frac{1}{N}\mathbf{h}_1^T \boldsymbol{\delta}_2, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}_2} = \frac{1}{N}\sum \boldsymbol{\delta}_2$$
$$\boldsymbol{\delta}_1 = (\boldsymbol{\delta}_2 \mathbf{W}_2^T) \odot \mathbf{h}_1 \odot (1 - \mathbf{h}_1)$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \frac{1}{N}\mathbf{X}^T \boldsymbol{\delta}_1, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \frac{1}{N}\sum \boldsymbol{\delta}_1$$

---

## Technical Specifications

| Hyperparameter / Dimension | Value | Design Rationale |
| :--- | :--- | :--- |
| **Input Features** | 30 Dimensions | Wisconsin Breast Cancer FNA cytology measurements |
| **Hidden Layer 1** | 128 Neurons | Captures non-linear cellular boundary feature maps |
| **Hidden Layer 2** | 128 Neurons | High-capacity representation layer for tumor morphology |
| **Output Layer** | 1 Neuron (Sigmoid) | Binary classification: 0 (Malignant) vs 1 (Benign) |
| **Training Epochs** | 1,000 Iterations | Convergence to stable minimum |
| **Learning Rate ($\alpha$)**| 0.01 | Gradient descent step size |
| **Test Set Split** | 20% (114 Patients) | Unseen evaluation holdout |
| **Framework Dependencies** | NumPy, Scikit-Learn (Dataset loading only) | Zero dependency on autograd frameworks |

---

## Empirical Benchmark Results

| Metric | Measured Value |
| :--- | :---: |
| **Test Accuracy** | **90.35%** |
| **Correct Classifications (Holdout)** | 103 / 114 Patients |
| **Optimization Convergence** | Monotonic loss reduction over 1,000 epochs |

---

## Project Structure

```
breast-cancer-nn-classifier/
├── ML.ipynb # Complete NumPy neural network implementation and training
├── requirements.txt # Environment dependencies
└── README.md # Technical documentation
```

---

## Installation and Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/codeblock1122/Breast-Cancer-NN-Classifier.git
cd Breast-Cancer-NN-Classifier
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
scikit-learn>=1.3.0
jupyter>=1.0.0
```

---

## Usage Guide

Execute the training notebook in Jupyter:
```bash
jupyter notebook ML.ipynb
```
