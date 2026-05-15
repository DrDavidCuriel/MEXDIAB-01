# MEXDIAB-01: Metabolic Disease Prediction in Mexico

Predictive modeling of dyslipidemia and myocardial infarction in the Mexican population using ENSANUT 2022 microdata. XGBoost classifiers with SHAP explainability and choropleth maps by state.

## Overview

This project applies machine learning to national health survey data (ENSANUT 2022) to identify key risk factors for two major metabolic conditions in Mexico: dyslipidemia and myocardial infarction. Results are visualized through SHAP summary plots and interactive choropleth maps at the state level.

## Features

- Binary classification models (XGBoost) for dyslipidemia and myocardial infarction
- SHAP-based explainability: summary plots, waterfall charts, and dependence plots
- Interactive choropleth maps by Mexican state
- Modular pipeline: data merging → feature engineering → training → explainability → visualization

## Project Structure
MEXDIAB-01/
├── src/
│   ├── 01_merge_clean.py          # Data merging and cleaning
│   ├── 02_feature_engineering.py  # Feature construction
│   ├── 03_model_training.py       # XGBoost training and evaluation
│   ├── 04_shap_analysis.py        # SHAP explainability
│   └── 05_choropleth_maps.py      # Interactive map generation
├── outputs/
│   ├── shap/                      # SHAP visualizations
│   ├── maps/                      # Choropleth HTML maps
│   ├── figures/                   # ROC and PR curves
│   └── models/                    # Trained model files (.pkl)
├── .gitignore
└── README.md

## Data Source

**ENSANUT 2022** (Encuesta Nacional de Salud y Nutrición)  
Instituto Nacional de Salud Pública — Mexico  
https://ensanut.insp.mx

> Raw microdata is not included in this repository per INSP data use guidelines.

## Tech Stack

- Python (pandas, scikit-learn, XGBoost, SHAP)
- Folium / Plotly for interactive maps
## Author

**David Curiel, MD**  
Medical Biotechnology | Clinical Data Science  
[LinkedIn](https://www.linkedin.com/in/david-alejandro-curiel-pedraza-6a918b241/) · [GitHub](https://github.com/DrDavidCuriel)
