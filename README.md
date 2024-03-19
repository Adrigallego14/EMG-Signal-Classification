# Gesture Classification using EMG Signals

This repository contains the code and documentation for my thesis project on the classification of gestures using Electromyography (EMG) signals.

## Overview

The project aims to develop a robust system for recognizing and classifying hand gestures based on EMG signal patterns. We utilized data recorded using Mindrove sensors from three male subjects performing eight distinct hand gestures, including the idle state. The dataset was collected under various conditions to ensure robustness.

## Methodology

**Plotting**: The `plot_Tasks.py` script provides functions to visualize EMG signals with colors corresponding to different tasks, aiding in data analysis.

**Signal Preprocessing**: Raw EMG signals are preprocessed using the `Preprocessing.py` script to remove noise and artifacts. Common preprocessing techniques such as filtering, normalization, and baseline correction are applied.

**Feature Extraction**: Relevant features are extracted from preprocessed signals using the `Feature_extraction.py` script. These features may include time-domain, frequency-domain, and statistical features.

**Classification Models**:
- `evaluate_knn.py`: Evaluates the performance of the k-Nearest Neighbors (kNN) classifier.
- `evaluate_mlp.py`: Evaluates the performance of the Multi-Layer Perceptron (MLP) classifier.
- `evaluate_rbf_svm.py`: Evaluates the performance of the Radial Basis Function (RBF) Support Vector Machine (SVM) classifier.

## Repository Structure

- `.gitignore`: Specifies intentionally untracked files to ignore.
- `Feature_extraction.py`: Script for feature extraction from EMG signals.
- `Preprocessing.py`: Script for preprocessing raw EMG signals.
- `evaluate_knn.py`: Script to evaluate kNN classifier performance.
- `evaluate_mlp.py`: Script to evaluate MLP classifier performance.
- `evaluate_rbf_svm.py`: Script to evaluate RBF SVM classifier performance.
- `main.py`: Main script to orchestrate the feature extraction, preprocessing, and classification process.
- `plot_Tasks.py`: Script containing functions to visualize results.
- `/data`: Contains the dataset used in the project.
- `/models`: Trained models saved for future use.

## Usage

To run the code:
1. Clone this repository: `git clone https://github.com/your_username/gesture-classification-emg.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Navigate to the `/src` directory and execute the desired scripts according to your requirements.

## Results

We present the results of our experiments, including classification accuracy, confusion matrices, and additional evaluation metrics.

## Future Work

- Explore advanced signal processing techniques to improve classification performance.
- Investigate the integration of deep learning models for gesture classification.
- Extend the dataset to enhance model generalization.

## Contributors

- [Adrián Gallego Mogena]
- [Edwin Daniel Oña Simbaña]


