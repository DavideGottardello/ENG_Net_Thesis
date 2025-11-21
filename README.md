# ENG_Net_Thesis
Thesis work analyzing ENG data from 3 rats (16 sciatic electrodes, 30 kHz). Classifies 4 stimulations (optional 5th rest) using scalable CNNs. Data are randomly generated; outputs compare performance across animals and 5-fold CV with boxplots and confusion matrices.
# Technical Documentation for GitHub Repository

This repository provides an example workflow for training and evaluating convolutional neural networks (CNNs) on EEG data. **Note:** Original data cannot be shared, so all data used here are randomly generated with the same shape as the real dataset. To properly evaluate all boxplots, the `main` script must be run at least three times, changing the animal each time.

---

## Repository Structure

* **`main.ipynb`** – Main workflow script that executes the entire process: data preparation, model training, validation, testing, and saving results.
* **`toolbox/`** – Contains all reusable functions and models:

  * **`models/EEGModels.py`** – EEG-specific neural network architectures (EEGNet variants).
  * **`models/CNNModels.py`** – CNN and ConvLSTM architectures.
  * **`functions/Functions.py`** – Data loading, preprocessing, encoding, normalization, and model creation utilities.
  * **`plots/Plot.py`** – Plotting functions for training history and results.

---

## Workflow Overview

### 1. Import Modules

Load Python libraries required for neural networks (TensorFlow/Keras), data manipulation (NumPy, Pandas), metrics (scikit-learn), and plotting (Matplotlib, Seaborn). GPU integration is optional and detected automatically. Toolbox modules are imported from `toolbox/models`, `toolbox/functions`, and `toolbox/plots`.

---

### 2. Set Parameters

Define all runtime parameters, including:

* Dataset and animal selection.
* Signal length and number of classes.
* Model selection (e.g., EEGNet variant).
* Cross-validation folds and train/test split.
* Optional normalization, signal cutting, and bias compensation.

Fixed parameters (e.g., number of electrodes, sampling frequency) are defined for Newcastle's dataset and rarely need modification.

---

### 3. Data Import and Preprocessing

* Define paths for dataset and output folders.
* Create directories for saving results and TensorBoard logs.
* Generate random data samples with the same shape as the original EEG data for demonstration purposes.
* Split the dataset into training+validation and test subsets.

---

### 4. Cross-Validation Setup

* Stratified K-Fold cross-validation is configured to ensure class balance across folds.
* Separate training and validation indices are generated for each fold.

---

### 5. Model Training

For each cross-validation fold:

* A folder is created to store results for the fold.
* Model architecture is created using the `create_model` function from the toolbox.
* Training is performed with early stopping and best-weight checkpointing.
* TensorBoard logs are created for visualization.
* Validation metrics (accuracy, F1-score, confusion matrix) are calculated and stored.

---

### 6. Testing

* Predictions are performed on the test set.
* Accuracy, F1-score, weighted F1, and confusion matrices are computed.
* Results are stored in a structured dictionary.

---

### 7. Results Saving and Plotting

* Training history and validation results are plotted using toolbox plotting functions.
* Cross-validation and test results are saved as a pickled dictionary.
* Validation and test metric arrays are exported as `.npz` files for boxplot visualization.

---

### Remarks

* This repository uses example random data for demonstration; real data is not included.
* To generate correct boxplots, the main workflow must be executed multiple times with different animals.
* All reusable models and functions are in the `toolbox/` folder.

---

This setup allows straightforward adaptation to real EEG datasets, replacing the random data generator with actual data loaders.
