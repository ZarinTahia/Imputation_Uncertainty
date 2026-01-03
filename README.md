# Imputation_Uncertainty

This repository contains implementations and experiments for **missing data imputation with uncertainty estimation**.  
It evaluates classical, optimization-based, generative, adversarial, and diffusion-based imputers under different missingness mechanisms.

---

## Overview

The goal of this project is to study the relationship between **imputation accuracy** and **uncertainty calibration**.  
We compare a diverse set of imputation methods and analyze how well their predicted uncertainties reflect true imputation errors across multiple datasets and missingness settings.

---

## Key Features

- Support for **MCAR, MAR, and MNAR** missingness  
- Unified evaluation pipeline across methods  
- Comparison of classical, generative, and diffusion-based imputers  
- Explicit evaluation of **uncertainty calibration** using ECE  
- Fully reproducible experiments via Jupyter notebooks  

---
## Project Structure

### Notebook/
- wine.ipynb – Wine Quality dataset
- BCancer.ipynb – Breast Cancer dataset
- Biodegradation.ipynb – Biodegradation dataset
- CaliforniaHousing.ipynb – California Housing dataset
- Energy.ipynb – Energy Efficiency dataset

### Output/

Saved results, metrics, and figures

### Core Scripts

- Inject_Missing_Values.py – MCAR, MAR, MNAR missingness injection
- imputers_updated.py – Unified imputation interface
- MIWAE.py – MIWAE and MIWAE-U implementations
- Gain.py – GAIN and GAIN-U implementations
- SoftImpute.py – Matrix factorization baseline
- TabCSDI.py – Diffusion-based tabular imputer
- OTImpute.py – Optimal Transport-based imputer
- Utils.py – Evaluation metrics and visualization utilities

### Other Files
- requirements.txt – Python dependencies
- README.md – Project documentation



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



@inproceedings{hossain2025imputation,
  title={Beyond Accuracy: An Empirical Study of Uncertainty Estimation in Imputation},
  author={Hossain, Zarin Tahia and Milani, Mostafa},
  year={2025},
  institution={Western University}
}

