# SAFIRE_ATR_wind_investigation

This project is focused on investigating wind sensing sytems (during the MAESTRO campaign). It includes statistical Physics based and data driven Machine Learning tools to qualify and correct wind measurements.

## Project Overview

- **Goal:** To analyze wind measurements from the SAFIRE ATR aircraft to better qualify them, calibrate sensors, and support ongoing research.
- **Data:** Requires the user to feed the data in a properly named folder (e.g Data/Raw/MAESTRO_IMU1). 
- **Main Tools:**  
  - LSTM and Attention-based Auto-encoders (Training and Usage)
  - Coarse segmentation algortithms
  - Planar fit method
  - Radome bias retrieval with optimization and ML tools

## Repository Structure

- `/Notebooks/` — Jupyter Notebooks for exploration and analysis.
- `/Data/` — Folder for data files used in the analysis.
- `/Plots/` — Folder for output figures.
- `README.md` — Project documentation.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LJaffeux/SAFIRE_ATR_wind_investigation.git
   ```
2. **Install dependencies:**  
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
   or install packages as needed, typically including `numpy`, `pandas`, `matplotlib`, `jupyter`...

3. **Run Jupyter Notebooks:**
   ```bash
   jupyter notebook
   ```
   Open any notebook in the `/notebooks/` directory to begin exploring the data.

# Notebooks Overview

This directory contains Jupyter Notebooks used in the investigation of wind measurements and calibration for the SAFIRE ATR campaigns. Each notebook is tailored to a specific step in the data processing, bias estimation, quality evaluation, or machine learning workflow.

## Notebook Descriptions

### 1. `Bias_estimation_PO.ipynb`
**Purpose:**  
Estimates biases in aircraft attack and sideslip angles (AOA/AOS) to ensure vertical wind (`W`) averages to zero at both flight and campaign scales. The notebook loads segmented and raw data, optimizes bias parameters via minimization, and visualizes the results for different campaigns and instrument units.

### 2. `Coarse_segmentation.ipynb`
**Purpose:**  
Performs segmentation of flight data into stable legs and maneuvers, using either pre-existing segmentation files or simple logic based on altitude, roll, and heading changes. This segmentation is a crucial preprocessing step for subsequent analysis (Planar fit and Roll, W correlations).

### 3. `Quality_evaluation.ipynb`
**Purpose:**  
Assesses the quality of wind measurements by applying planar fits to stabilized flight legs and analyzing roll/wind correlations outside those legs. Includes routines to systematically alter attitude corrections and evaluate their impact on derived wind statistics, supporting calibration and validation efforts.

### 4. `Train_Anomaly_detection_auto_encoder.ipynb`
**Purpose:**  
Trains LSTM-CNN hybrid and Attention-based autoencoders to detect anomalies in aircraft time series data. The notebook prepares datasets, configures and fits the model, and saves the trained model for later use in anomaly detection.

### 5. `Train_bias_estimator.ipynb`
**Purpose:**  
Trains a regression model (LSTM-based) to predict biases in AOA and AOS from time series data. Random biases are injected to form the training set. The notebook handles data preprocessing, model training, and evaluation.

### 6. `Use_Anomaly_detection_auto_encoder.ipynb`
**Purpose:**  
Loads a trained autoencoder model and applies it to new datasets for anomaly detection. Supports the selection of model versions, visualizes configuration and model architecture, and generates relevant plots to assess anomaly scores and feature contributions.

### 7. `Use_bias_estimator.ipynb`
**Purpose:**  
Loads and applies a trained bias regressor to flight data to estimate and visualize AOA/AOS biases across flights. Produces time series and summary plots of estimated biases and relates results to flight parameters such as altitude and heading.

---

*For more details on each notebook, see the markdown cells at the top of each file or consult the main repository README.*
