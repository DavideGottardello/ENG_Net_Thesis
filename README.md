This repository is part of my thesis work, analyzing ENG data recorded from three rats. Each signal comes from 16 electrodes positioned on the sciatic nerve, sampled at 30 kHz. Four types of stimulation were recorded, with an optional fifth class for resting. The goal is to perform scalable CNN classification, testing different network architectures to classify the recorded classes. The data provided here are randomly generated to match the original data structure, since the original recordings cannot be shared. Outputs include performance comparisons across different animals and folds, using 5-fold cross-validation, with boxplots and confusion matrices.

This repository provides an example workflow for training and evaluating convolutional neural networks (CNNs) on EEG data. **Note:** Original data cannot be shared, so all data used here are randomly generated with the same shape as the real dataset. To properly evaluate all boxplots, the `main` script must be run at least three times, changing the animal each time.

---
# Technical Documentation

This repository provides an example workflow for training and evaluating convolutional neural networks (CNNs) on ENG-like data. **Original data I used cannot be shared**, so all inputs are randomly generated with the same structure as the real dataset. To obtain the complete boxplot comparison, the `main` workflow must be run three times, changing the animal each time, and window size of the signals: 50, 100, 200, 500 ms.

---

## Repository Structure

### **1. `main.ipynb` – Core Workflow**

The central notebook that performs the full processing pipeline:

* Imports all required libraries and toolbox modules.
* Sets user-defined and fixed parameters.
* Generates random data shaped like the original ENG recordings.
* Splits samples into train/validation/test sets.
* Creates cross-validation folds.
* Trains selected CNN/EEGNet architectures.
* Evaluates performance on validation and test sets.
* Saves results (metrics, histories, predictions) and per-fold outputs.
* Stores arrays needed to compute boxplots and confusion matrices.

All model definitions and utility functions used here come from the `toolbox` directory.

---

### **2. `boxplots.ipynb` – Performance Comparison**

This notebook loads the `.npz` files produced by `main.ipynb` and:

* Builds accuracy and F1-score boxplots across folds.
* Compares results between different animals and different model architectures.

It requires that results for each animal have been previously generated.

---

### **3. `confusion_matrix.ipynb` – Confusion Analysis**

This notebook loads the stored predictions and:

* Computes confusion matrices for validation or test sets.
* Visualizes class-balanced and normalized matrices.
* Allows comparison across folds and animals.

---

## Workflow Overview

### **Import Modules**

Load scientific and deep-learning libraries, detect GPU capability, and import all models and functions from the `toolbox` directory:

* `toolbox/models` contains EEGNet and CNN architectures.
* `toolbox/functions` provides data loading, preprocessing, encoding, and model creation utilities.
* `toolbox/plots` offers plotting functions for training history and fold-wise metrics.

---

### **Parameter Setup**

Define:

* Dataset/animal identifiers
* Signal length and sampling parameters
* Number of classes
* Model architecture selection
* Cross-validation setup
* Output paths for TensorBoard logs and result files

Fixed parameters are included for the original dataset structure but can be modified for new data sources.

---

### **Data Import and Preprocessing**

* Create folders for results and logs.
* Generate example random data matching the shape of the real ENG dataset.
* Prepare samples, derive feature dimensions, and split train/validation/test sets.

---

### **Cross-Validation and Training**

* Apply stratified K-fold splitting.
* Train the chosen model with early stopping and checkpointing.
* Evaluate each fold and store metrics: accuracy, F1-scores, and confusion matrices.
* Save per-fold histories and results to structured output files.

---

### **Testing and Results Export**

* Evaluate performance on the held-out test set.
* Save all results in pickle files.
* Export validation/test metric arrays for boxplots and confusion matrices.

---

### Remarks

* Random data are used only to illustrate the workflow; real ENG signals are not included.
* Run `main.ipynb` multiple times (one per animal) to enable proper comparison in `boxplots.ipynb` and `confusion_matrix.ipynb`.
* All reusable functionality is stored in the `toolbox` folder for modularity and scalability.
