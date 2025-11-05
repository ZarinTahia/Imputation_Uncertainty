# Imputation_Uncertainty

This repository contains implementations and experiments for **missing data imputation with uncertainty estimation**.  
It includes several classical and deep generative imputation models and evaluates their performance across multiple datasets and missingness mechanisms (MCAR, MAR, MNAR).

---

## Overview

The goal of this project is to explore the relationship between **accuracy** and **uncertainty calibration** in data imputation.  
It implements and compares different imputation algorithms — from optimization-based to deep generative methods — and studies how well their predicted uncertainties align with true imputation errors.

---

## Project Structure

IMPUTATION_UNCERTAINTY/
│
├── Data/ # Datasets used for experiments
├── Notebook/ # Jupyter notebooks (one per dataset)
│ ├── BCancer.ipynb # Breast Cancer dataset
│ ├── Biodegradation.ipynb # Biodegradation dataset
│ ├── CaliforniaHousing.ipynb# California Housing dataset
│ ├── Disease.ipynb # Diabetes dataset
│ ├── Energy.ipynb # Energy Efficiency dataset
│ ├── GasSensor.ipynb # Gas Sensor Array Drift dataset
│ ├── HIGGS.ipynb # HIGGS dataset
│ ├── wine.ipynb # Wine Quality dataset
│ ├── experiment.ipynb # Main experiment pipeline
│ └── ex.ipynb, ch.ipynb... # Miscellaneous exploratory notebooks
│
├── Output/ # Saved results, metrics, and plots
│
├── Inject_Missing_Values.py # Generates MCAR, MAR, MNAR missingness
├── imputers_updated.py # Unified wrapper for all imputers
├── MIWAE.py # MIWAE and MIWAE-U (uncertainty-aware variant)
├── Gain.py # GAIN and GAIN-U implementations
├── SoftImpute.py # SoftImpute (matrix factorization) baseline
├── TabCSDI.py # TabCSDI diffusion-based imputer
├── Utils.py # Metrics, evaluation, and visualization utilities
├── OTImpute.py (if applicable) # Optimal Transport-based imputer
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## Implemented Methods

| Category | Algorithm | Description |
|-----------|------------|-------------|
| **Classical** | **MICE** | Multiple Imputation by Chained Equations; regression-based iterative imputer providing multiple plausible imputations. |
| **Matrix Factorization** | **SoftImpute** | Low-rank matrix completion via nuclear norm minimization. |
| **Optimal Transport** | **OT-Impute** | Imputation using Sinkhorn-based optimal transport minimization between observed and reconstructed distributions. |
| **Generative** | **MIWAE** | Importance-weighted autoencoder that learns a probabilistic latent representation for missing data. |
| **Adversarial** | **GAIN** | GAN-based imputation using a discriminator-guided reconstruction loss. |
| **Diffusion-based** | **TabCSDI** | Conditional diffusion model for tabular data imputation with uncertainty estimation. |

---

## Datasets

Experiments are performed on diverse **numeric tabular datasets**:

- **Wine Quality**
- **Breast Cancer**
- **Biodegradation**
- **California Housing**
-  **Energy Efficiency**

Each dataset notebook in `/Notebook/` (e.g., `wine.ipynb`, `Energy.ipynb`) executes the full pipeline:
1. Inject missing values (MCAR / MAR / MNAR)  
2. Apply all imputation methods (MICE, OT-Impute, SoftImpute, MIWAE, GAIN, TabCSDI)  
3. Evaluate **MAE**, **RMSE**, and **ECE** metrics  
4. Save results and calibration plots under `Output/`

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/Imputation_Uncertainty.git
cd Imputation_Uncertainty
pip install -r requirements.txt


Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/Imputation_Uncertainty.git
cd Imputation_Uncertainty
pip install -r requirements.txt

@inproceedings{hossain2025imputation,
  title={Beyond Accuracy: An Empirical Study of Uncertainty Estimation in Imputation},
  author={Hossain, Zarin Tahia and Milani, Mostafa},
  year={2025},
  institution={Western University}
}

