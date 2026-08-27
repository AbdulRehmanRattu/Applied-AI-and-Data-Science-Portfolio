
# Machine Learning Models for Handwritten Digit Recognition and Face Detection

This repository contains the implementation of three different machine learning models - Perceptron, a two-layer Neural Network, and Naive Bayes - to perform classification tasks such as recognizing handwritten digits and detecting faces in images.

## Project Overview

The main objective was to experiment with both simple and complex models to understand their learning and generalization capabilities on optical character recognition (OCR) and face detection tasks.

## Table of Contents
- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Model Implementation](#model-implementation)
 - [Perceptron](#perceptron)
 - [Neural Network](#neural-network)
 - [Naive Bayes](#naive-bayes)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [Contact](#contact)

## Introduction

The project involves the design and implementation of three different machine learning models to perform two classification tasks: recognizing handwritten digits and detecting faces in images. The models were trained and tested on separate datasets comprising scanned images of handwritten digits and pre-processed images where facial edges have been detected.

## Data Preparation

The data consists of:
- Handwritten digits from 0 to 9.
- Edge-detected images of faces.

The images were split into training, validation, and testing sets. Each model was trained progressively on 10% to 100% of the training data to observe scalability and performance improvements.

## Model Implementation

### Perceptron

The Perceptron was implemented using the standard learning rule, where weights were updated based on prediction errors.

### Neural Network

The Neural Network is a two-layer network with one hidden layer, utilizing ReLU and softmax activation functions.

### Naive Bayes

The Naive Bayes model employed probabilistic reasoning for classification based on Bayes' Theorem, assuming feature independence.

## Results

### Naive Bayes

- **Digits**: Accuracy improved from 69.1% to 76.3% as training data increased.
- **Faces**: Accuracy started at 66.0% and peaked at 94.0%.

### Neural Network

- **Digits**: Started at 76.4% accuracy with 500 training samples and reached 84.7% with 5000 samples.
- **Faces**: Accuracy ranged from 71.3% to 93.3%, with some issues noted regarding potential overfitting or data anomalies.

### Perceptron

- **Digits**: High variability in performance with accuracy peaking at 95.1%.
- **Faces**: Consistently improved performance, ending at 90.7% accuracy.

## Discussion

The results indicate that while simple models like Perceptron can perform remarkably well on digit classification, more complex models like Neural Networks are required to handle the nuances of face detection effectively. However, the Neural Network's performance on faces showed potential issues related to training stability and model configuration, suggesting a need for further parameter tuning and possibly more sophisticated regularization techniques.

## Conclusion

This project demonstrated the application of foundational machine learning concepts to practical classification tasks. Each model offered unique insights into the challenges and complexities of pattern recognition. The progressive training approach provided valuable data on how models respond to increasing training data sizes, informing future work on model tuning and selection.

## Usage

1. **Clone the repository:**
 ```sh
 git clone https://github.com/AbdulRehmanRattu/Machine-Learning-Models-for-Handwritten-Digit-Recognition-and-Face-Detection.git
 cd Machine-Learning-Models-for-Handwritten-Digit-Recognition-and-Face-Detection
 ```

2. **Run the models:**
 - For Naive Bayes on Digits:
 ```sh
 python NaiveBayes_Digits.py
 ```
 - For Naive Bayes on Faces:
 ```sh
 python NaiveBayes_Faces.py
 ```
 - For Neural Network on Digits:
 ```sh
 python Neural_Network_Digits.py
 ```
 - For Neural Network on Faces:
 ```sh
 python Neural_Network_Faces.py
 ```
 - For Perceptron on Digits:
 ```sh
 python Perceptron_Digits.py
 ```
 - For Perceptron on Faces:
 ```sh
 python Perceptron_Faces.py
 ```

## Contact

For any inquiries, please contact me at:

- **Email:** [rattu786.ar@gmail.com](mailto:rattu786.ar@gmail.com)
- **LinkedIn:** [Abdul Rehman Rattu](https://www.linkedin.com/in/abdul-rehman-rattu-395bba237)
