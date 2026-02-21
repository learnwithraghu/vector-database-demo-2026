# 📈 Vector Visualization

This directory contains the `02_visualization.ipynb` notebook. It was separated from the main Streamlit application quickstart to keep the main application clean and focused solely on the user interface. 

## Purpose
The notebook is an advanced module that demonstrates how to extract vectors from LanceDB and project the high-dimensional data (384 dimensions) down to 2D space using PCA (Principal Component Analysis) for visualization in Plotly. This helps students visualize the semantic clustering of the 5-year shareholder letter corpus.

## Requirements
If you wish to run this notebook locally, you will need a Python environment with:
- `lancedb`
- `pandas`
- `scikit-learn`
- `plotly`
- `ipykernel`

And you must point it to a valid LanceDB database containing embeddings (such as the one generated in `00-Quickstart`).
