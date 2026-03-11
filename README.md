# Time Series Pattern Classification

This project is an interview assignment for binary classification on temporal data.

## Objective
The goal is to detect repeating patterns present in class 1 sequences from time-series samples of shape `(batch, 10, 2)`.

## Project Structure
- `data/` : local dataset files (`X.npz`, `y.npz`) - not pushed to GitHub
- `notebooks/` : Jupyter notebook analysis
- `src/` : helper scripts if needed
- `reports/` : exported figures or outputs

## Data
The dataset files are not included in this repository due to size constraints.
Place `X.npz` and `y.npz` inside the `data/` folder before running the notebook.

## Main Steps
- data inspection
- pattern discovery
- feature engineering
- model training
- model selection
- model evaluation
- production considerations