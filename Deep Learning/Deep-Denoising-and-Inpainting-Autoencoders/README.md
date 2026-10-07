# Deep Denoising and Blind Inpainting Autoencoders: Multi-Corruption Image Restoration

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Domain: Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-purple.svg)]()
[![Model: Convolutional Autoencoder](https://img.shields.io/badge/Model-Convolutional%20Autoencoder-0284C7.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An unsupervised deep learning framework and contractive convolutional autoencoder suite designed to restore degraded images subjected to complex real-world corruption scenarios: high-variance additive Gaussian noise, blind spatial occlusion inpainting, and combined multi-corruption degradation.

By enforcing a 64x spatial downsampling bottleneck $(32 \times 32 \times 3 \rightarrow 4 \times 4 \times 128)$, the architecture learns topological manifold projections that discard high-frequency stochastic noise while inpainting missing semantic image regions.

---

## Scientific Problem Formulation

Image corruption degrades natural pixel manifolds through stochastic perturbations and physical occlusion:

### 1. Additive Gaussian Noise Denoising
$$\tilde{x}_{\text{gauss}} = \text{clip}\big(x + \eta, 0.0, 1.0\big), \quad \eta \sim \mathcal{N}(0, \sigma^2 \mathbf{I}), \quad \sigma = 0.20$$

### 2. Blind Spatial Occlusion Inpainting
$$\tilde{x}_{\text{occluded}} = x \odot \mathcal{M}, \quad \mathcal{M} \in \{0, 1\}^{32 \times 32}$$
Where an $8 \times 8$ rectangular patch is dropped at arbitrary spatial coordinates without location cues.

### 3. Combined Multi-Corruption Restoration
$$\tilde{x}_{\text{combined}} = \text{clip}\big(x \odot \mathcal{M} + \eta, 0.0, 1.0\big)$$

---

## End-to-End System Architecture

<p align="center">
  <img src="docs/assets/autoencoder_architecture.png" alt="Autoencoder Architecture" width="95%"/>
</p>

### Detailed Layer Specification

| Stage | Block Type | Filters | Kernel | Strides | Activation | Output Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | Input Image | - | - | - | - | $32 \times 32 \times 3$ |
| **Encoder** | Conv Block 1 | 32 | $3 \times 3$ | $1 \times 1$ | ReLU | $32 \times 32 \times 32$ |
| **Encoder** | MaxPool 1 | - | $2 \times 2$ | $2 \times 2$ | - | $16 \times 16 \times 32$ |
| **Encoder** | Conv Block 2 | 64 | $3 \times 3$ | $1 \times 1$ | ReLU | $16 \times 16 \times 64$ |
| **Encoder** | MaxPool 2 | - | $2 \times 2$ | $2 \times 2$ | - | $8 \times 8 \times 64$ |
| **Encoder** | Conv Block 3 | 128 | $3 \times 3$ | $1 \times 1$ | ReLU | $8 \times 8 \times 128$ |
| **Bottleneck**| Latent Pool | - | $2 \times 2$ | $2 \times 2$ | - | $4 \times 4 \times 128$ (2,048 dims) |
| **Decoder** | UpSample 1 | - | $2 \times 2$ | - | - | $8 \times 8 \times 128$ |
| **Decoder** | Conv Block 1 | 64 | $3 \times 3$ | $1 \times 1$ | ReLU | $8 \times 8 \times 64$ |
| **Decoder** | UpSample 2 | - | $2 \times 2$ | - | - | $16 \times 16 \times 64$ |
| **Decoder** | Conv Block 2 | 32 | $3 \times 3$ | $1 \times 1$ | ReLU | $16 \times 16 \times 32$ |
| **Decoder** | UpSample 3 | - | $2 \times 2$ | - | - | $32 \times 32 \times 32$ |
| **Output** | Output Conv | 3 | $3 \times 3$ | $1 \times 1$ | Sigmoid | $32 \times 32 \times 3$ |

---

## Empirical Benchmark & Performance Results

All models were evaluated across test distributions using sample-averaged Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM):

<p align="center">
  <img src="docs/assets/performance_metrics_benchmark.png" alt="Performance Benchmark" width="90%"/>
</p>

| Corruption Modality | Test Loss (MSE) | Reconstruction MSE ($\times 10^{-3}$) | PSNR (dB, Higher is Better) | SSIM Score (Higher is Better) | Optimal Epoch |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Spatial Inpainting (8x8 Occlusion)** | 0.0062 | **6.20** | **22.08 dB** | **0.885** | Epoch 38 |
| **Gaussian Denoising ($\sigma=0.20$)** | 0.0081 | **8.10** | **20.91 dB** | **0.842** | Epoch 41 |
| **Combined Multi-Corruption** | 0.0122 | **12.20** | **19.14 dB** | **0.778** | Epoch 44 |

### Convergence Dynamics
Convergence trajectories across 50 training epochs illustrate smooth mini-batch optimization with early validation plateauing:

<p align="center">
  <img src="docs/assets/training_convergence_curves.png" alt="Training Convergence Curves" width="95%"/>
</p>

---

## Visual Reconstruction Demonstrations

### 1. Gaussian Denoising ($\sigma = 0.20$)
<p align="center">
  <img src="docs/assets/denoising_reconstruction_grid.png" alt="Gaussian Denoising Grid" width="95%"/>
</p>

### 2. Blind Spatial Inpainting ($8 \times 8$ Occlusion)
<p align="center">
  <img src="docs/assets/inpainting_reconstruction_grid.png" alt="Spatial Inpainting Grid" width="95%"/>
</p>

### 3. Combined Multi-Corruption Restoration
<p align="center">
  <img src="docs/assets/combined_reconstruction_grid.png" alt="Combined Multi-Corruption Grid" width="95%"/>
</p>

---

## Project Structure

```bash
Deep-Denoising-and-Inpainting-Autoencoders/
├── autoencoder_restoration/
│   ├── __init__.py                     # Package exports
│   ├── corruptions.py                  # Gaussian noise and spatial occlusion generators
│   ├── data_loader.py                  # Structured dataset and CIFAR-10 ingestion
│   ├── metrics.py                      # MSE, PSNR, and SSIM evaluators
│   ├── trainer.py                      # Training engine and loss tracker
│   └── models/
│       ├── __init__.py
│       └── convolutional_autoencoder.py # Keras and PyTorch autoencoder architectures
├── docs/
│   ├── assets/
│   │   ├── autoencoder_architecture.png       # 300 DPI system architecture
│   │   ├── combined_reconstruction_grid.png   # 300 DPI multi-corruption visual grid
│   │   ├── denoising_reconstruction_grid.png  # 300 DPI Gaussian denoising grid
│   │   ├── inpainting_reconstruction_grid.png # 300 DPI spatial inpainting grid
│   │   ├── performance_metrics_benchmark.png  # 300 DPI comparative bar chart
│   │   └── training_convergence_curves.png    # 300 DPI 50-epoch loss curves
│   ├── Deep_Denoising_and_Inpainting_Autoencoders_Report.docx # Word technical report
│   └── RESEARCH_REPORT.md                     # Markdown monograph with cover page
├── notebooks/
│   └── BEST_VERSION.ipynb              # Historical interactive notebook
├── generate_visuals.py                 # 300 DPI visual asset generator
├── run_pipeline.py                     # Unified CLI driver
├── requirements.txt                    # Project dependencies
├── LICENSE                             # MIT License
└── README.md                           # Documentation
```

---

## Quickstart & CLI Execution

### 1. Environment Setup
```bash
cd "Deep Learning/Deep-Denoising-and-Inpainting-Autoencoders"
pip install -r requirements.txt
```

### 2. Run Single Corruption Restorations
```bash
# Gaussian Denoising
python3 run_pipeline.py --mode gaussian --epochs 10

# Spatial Inpainting
python3 run_pipeline.py --mode occlusion --epochs 10

# Combined Multi-Corruption
python3 run_pipeline.py --mode combined --epochs 10
```

### 3. Run Benchmark Suite
```bash
python3 run_pipeline.py --mode benchmark --epochs 5
```

### 4. Regenerate 300 DPI Publication Visuals
```bash
python3 generate_visuals.py
```

---

## Python API Usage

```python
from autoencoder_restoration import (
    load_and_preprocess_dataset,
    AutoencoderTrainer,
    evaluate_restoration_batch
)

# 1. Load normalized benchmark image dataset
x_train, x_test = load_and_preprocess_dataset(subset_train_size=2000, subset_test_size=500)

# 2. Instantiate and train denoising autoencoder
trainer = AutoencoderTrainer(corruption_type="gaussian", noise_factor=0.20)
trainer.fit(x_train, x_val=x_test[:100], epochs=10, batch_size=128)

# 3. Evaluate restoration fidelity
metrics = trainer.evaluate(x_test)
print(f"Test MSE:  {metrics['mse']:.5f}")
print(f"Test PSNR: {metrics['psnr_db']:.2f} dB")
print(f"Test SSIM: {metrics['ssim']:.4f}")
```

---

## Technical Report Reference

Detailed formal research documentation is available in:
- [Interactive Technical Monograph (Markdown)](docs/RESEARCH_REPORT.md)
- [Formal Research Report (Word Document with Cover Page)](docs/Deep_Denoising_and_Inpainting_Autoencoders_Report.docx)

---

## Author & Citation

Built and maintained by **Abdul Rehman Rattu** (Forward Deployed AI Engineer & Solutions Architect).

```bibtex
@software{rattu2026deepautoencoders,
  author = {Abdul Rehman Rattu},
  title = {Deep Denoising and Blind Inpainting Autoencoders: Multi-Corruption Image Restoration},
  year = {2026},
  publisher = {GitHub},
  journal = {Applied AI and Data Science Portfolio},
  howpublished = {\url{https://github.com/AbdulRehmanRattu/Applied-AI-and-Data-Science-Portfolio}}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
