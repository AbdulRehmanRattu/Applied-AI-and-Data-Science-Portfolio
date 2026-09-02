# Applied AI and Data Science Master Portfolio and Learning Roadmap

<div align="center">

[<img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">](https://opensource.org/licenses/Apache-2.0)
<img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Vision-PyTorch%20%7C%20OpenCV-EE4C2C.svg?logo=pytorch&logoColor=white" alt="Vision">
<img src="https://img.shields.io/badge/API-Flask-0052FF.svg?logo=flask&logoColor=white" alt="Flask">
<img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg" alt="Status">

**Enterprise-grade, high-performance implementation built and maintained by Abdul Rehman Rattu.**

[Overview](#overview) • [Key Features](#key-features) • [Installation & Usage](#quickstart--usage) • [Author & Maintainer](#author--maintainer)

</div>

---

A comprehensive, hands-on open-source curriculum and reference portfolio designed for **students, aspiring data scientists, and AI engineers** to learn, practice, and master real-world Machine Learning, Deep Learning, Computer Vision, Generative AI, and Intelligent Systems.

Curated and maintained by **Abdul Rehman Rattu** (Founder and CEO at [Rapide Technologies]()).

---

## Mission and Purpose

The goal of this repository is to bridge the gap between academic theory and practical implementation. Whether you are a student starting your data science journey or an experienced engineer sharpening your applied skills, this repository provides:

- Real-world problem statements and end-to-end solutions.
- Well-documented Jupyter notebooks with verified outputs, diagnostic plots, and benchmark tables.
- A progressive 3-level learning path (Fundamentals to Advanced Production Engineering).
- Ready-to-run environments with modern dependencies.

---

## How to Use This Repository for Learning and Practice

1. **Star and Fork**: Click the **Star** and **Fork** buttons at the top right to save this roadmap to your GitHub and practice independently.
2. **Clone the Repository**:
   ```bash
   git clone https://github.com/AbdulRehmanRattu/Applied-AI-and-Data-Science-Portfolio.git
   cd Applied-AI-and-Data-Science-Portfolio
   ```
3. **Set Up the Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Follow the Progressive Learning Path**:
   - Start with **Level 1 (Fundamentals)** to master exploratory data analysis, data cleaning, and core Scikit-Learn workflows.
   - Advance to **Level 2 (Intermediate)** to build computer vision filters, NLP assistants, and non-linear ensemble models.
   - Conquer **Level 3 (Advanced)** to write neural networks from scratch using pure matrix calculus, build generative models (DCGAN), and deploy full-stack AI web and mobile apps.

---

## Learning Path and Competency Matrix

### Level 1: Fundamentals

Master core data manipulation, feature engineering, exploratory data analysis, and baseline classification and regression workflows.

| # | Project | What You Will Learn | Category | Verified Metric / Outcome |
| :---: | :--- | :--- | :--- | :--- |
| 1 | [Red Wine Quality Prediction](Classification/Red-Wine-Quality-Prediction/) | Physicochemical EDA, outlier analysis, Random Forest vs. SVM benchmarking, class weight balancing. | Classification | Tuned Random Forest: **67.81% Acc**, Macro F1: 0.61 (8 diagnostic plots). |
| 2 | [Mammal Sleep Duration Forecasting](Regression/Mammal-Sleep-Duration-Forecasting/) | Allometric biological power-law scaling, winsorization, multivariate OLS vs. XGBoost gradient boosting. | Regression | XGBoost: **MSE 0.1338**, RMSE 0.3658 vs. OLS MSE 0.6686 (8 diagnostic plots). |
| 3 | [Tic-Tac-Toe Adversarial Minimax](Game%20AI%20%26%20Reinforcement%20Learning/TicTacToe-Adversarial-Minimax/) | Zero-sum game theory, recursive state search, terminal state utility evaluation, Pygame desktop GUI. | Game AI | Mathematically unbeatable (zero loss rate across 5,478 reachable states). |

---

### Level 2: Intermediate and Diagnostic Systems

Develop robust diagnostic pipelines, non-linear ensemble models, computer vision filtering, and Large Language Model integrations.

| # | Project | What You Will Learn | Category | Verified Metric / Outcome |
| :---: | :--- | :--- | :--- | :--- |
| 4 | [Predictive Models for Diabetes Detection](Classification/Predictive-Models-for-Diabetes-Detection/) | Large-scale electronic health record (100k records) stratification, 3-fold CV, Random Forest, SVM, ANN. | Classification | Random Forest: **97.18% CV Accuracy**, 0.95 Precision, 0.80 F1-Score (3 confusion matrix plots). |
| 5 | [Cardiovascular Heart Disease Risk (KNN)](Classification/Cardiovascular-Heart-Disease-Risk-KNN/) | Distance-metric optimization (Manhattan vs. Euclidean), inverse-distance weighting, 10-fold CV tuning. | Classification | Distance-Weighted Manhattan KNN ($K=9$): **99.71% (+-0.87%) CV Accuracy** (4 plots). |
| 6 | [Airbnb Nightly Price Valuation](Regression/Airbnb-Nightly-Price-Prediction/) | High-cardinality geographic encoding, multi-hot amenity tokenization, gradient boosting price modeling. | Regression | Gradient Boosting: **Log-Price MSE 0.1689**, RMSE 0.4110, $R^2$ 0.7420 (2 plots). |
| 7 | [Automated Medical X-Ray Enhancement](Computer%20Vision/Automated-Medical-X-Ray-Image-Enhancement/) | Pre-classification computer vision filter chain (MAD noise estimation, CLAHE, morphological inpainting). | Computer Vision | Upstream enhancement improves pneumonia classifier accuracy from **55.0% to 95.2% (+40.2%)**. |
| 8 | [Microscopic Specimen Motion Tracker](Computer%20Vision/Microscopic-Specimen-Motion-Tracker/) | Multi-threaded OpenCV video stream processing, background subtraction, motility contour kinematics. | Computer Vision | Real-time 30+ FPS biological specimen velocity tracking and PyInstaller standalone build. |
| 9 | [Connect Four Minimax Alpha-Beta](Game%20AI%20%26%20Reinforcement%20Learning/Connect4-Minimax-AlphaBeta-Engine/) | Depth-limited adversarial search, Alpha-Beta branch pruning, sliding-window heuristic board scoring. | Game AI | Depth-5 search with 99.4% state tree pruning and Pygame desktop interface. |
| 10 | [AI Resume Scoring Assistant (GPT-3.5)](NLP%20%26%20Generative%20AI/AI-Resume-Scoring-Assistant-GPT35/) | PDF/DOCX document parsing, OpenAI API prompt chaining, structured candidate scoring, Tkinter UI. | NLP & GenAI | Automated candidate evaluation across core skills, prerequisite criteria, and match score. |
| 11 | [AstroAI Conversational Desktop Assistant](NLP%20%26%20Generative%20AI/AstroAI-Conversational-Desktop-Assistant/) | Ephemeris planetary coordinate calculation, geometric aspect engines, rule-based astrological dialogue. | NLP & GenAI | Interactive Tkinter application with real-time astronomical transit calculations. |

---

### Level 3: Advanced and Production Engineering

Build from-scratch deep neural networks with matrix calculus, generative adversarial vision, multi-century recurrent sequence forecasting, full-stack web applications, and native mobile platforms.

| # | Project | What You Will Learn | Category | Verified Metric / Outcome |
| :---: | :--- | :--- | :--- | :--- |
| 12 | [Breast Cancer Custom Neural Network](Deep%20Learning/Breast-Cancer-Custom-Neural-Network/) | From-scratch multi-layer perceptron in pure NumPy, explicit matrix calculus backpropagation derivatives. | Deep Learning | Pure NumPy MLP: **90.35% Holdout Accuracy**, vectorized batch gradient descent (1000 epochs). |
| 13 | [Handwritten Digit and Face Classifiers](Computer%20Vision/Handwritten-Digit-and-Face-Classifiers/) | Mathematical implementation of Linear Perceptron, Bernoulli Naive Bayes, and MLP from first principles. | Computer Vision | Digits: **84.0%** (Perceptron), **83.0%** (Naive Bayes), **87.0%** (MLP). Faces: **87.0%** (Perceptron). |
| 14 | [Credit Card Fraud Detection Flask App](Classification/Credit-Card-Fraud-Detection-Flask/) | 99.4% class imbalance mitigation, SMOTE resampling, high-throughput fraud scoring, Flask web UI. | Classification | Fraud Classifier: **96.60% Accuracy**, 0.97 Precision, 0.96 Recall, **0.98 ROC-AUC** + Web App. |
| 15 | [DCGAN Synthesis and SqueezeNet Vision](Deep%20Learning/DCGAN-Synthesis-and-SqueezeNet-Vision/) | PyTorch DCGAN (50,000 steps), SLERP latent manifold interpolation, Fire-module SqueezeNet edge classifier. | Deep Learning | DCGAN photorealistic synthesis + edge-optimized SqueezeNet classifier (<4.8MB footprint). |
| 16 | [Global Climate Temperature Forecasting](Time%20Series/Global-Climate-Temperature-Forecasting/) | Multi-century sensor data harmonization (1750 to present), seasonal ARIMA decomposition, Deep LSTM. | Time Series | ARIMA + Deep LSTM planetary climate forecasting + GeoPandas agricultural yield regression. |
| 17 | [Smart GP Community Android App](Mobile%20%26%20Systems%20Engineering/Smart-GP-Community-Android-Telehealth-App/) | Native Android MVVM architecture, Kotlin Coroutines, Jetpack components, clean service layers. | Mobile Engineering | Production-ready Android telehealth application (doctor directory, appointment booking, auth). |

---

## Categories Overview

Projects are organized into domain-specific category directories, each containing self-contained implementations, datasets, verified execution plots, and dedicated technical documentation:

- [Classification](Classification/) (4 Projects)
- [Regression](Regression/) (2 Projects)
- [Time Series](Time%20Series/) (1 Project)
- [Computer Vision](Computer%20Vision/) (3 Projects)
- [Deep Learning](Deep%20Learning/) (2 Projects)
- [NLP and Generative AI](NLP%20%26%20Generative%20AI/) (2 Projects)
- [Game AI and Reinforcement Learning](Game%20AI%20%26%20Reinforcement%20Learning/) (2 Projects)
- [Mobile and Systems Engineering](Mobile%20%26%20Systems%20Engineering/) (1 Project)

---

## Technical Stack

| Domain | Frameworks and Libraries |
| :--- | :--- |
| **Languages** | Python 3.8+, Kotlin, SQL, JavaScript |
| **Machine Learning** | Scikit-Learn, XGBoost, Statsmodels, SciPy |
| **Deep Learning** | PyTorch, Torchvision, TensorFlow, Keras, NumPy (From Scratch) |
| **Computer Vision** | OpenCV, PIL, Scikit-Image, Matplotlib, Seaborn |
| **NLP and LLMs** | OpenAI API (GPT-3.5 Turbo), PyPDF2, python-docx |
| **Web and APIs** | Flask, RESTful APIs, Jinja2, HTML5, CSS3 |
| **Mobile and Desktop** | Native Android (Kotlin MVVM, Gradle DSL), Tkinter, Pygame |
| **Geospatial Analytics** | GeoPandas, Folium, Shapefile GIS |

---

## Community and Contributions

Contributions, bug reports, and pull requests from students, developers, and educators are warmly welcomed. If you build upon any of these projects or use them in your learning journey, feel free to submit a pull request or share your findings!

---

---

## Author & Maintainer

**Abdul Rehman Rattu**  
*Forward Deployed AI Engineer & Solutions Architect*  
*Founder & Technical Lead, Rapide Technologies*

* **Email**: [rattu786.ar@gmail.com](mailto:rattu786.ar@gmail.com)
* **LinkedIn**: [linkedin.com/in/abdul-rehman-rattu-395bba237](https://www.linkedin.com/in/abdul-rehman-rattu-395bba237)
* **GitHub**: [github.com/AbdulRehmanRattu](https://github.com/AbdulRehmanRattu)
