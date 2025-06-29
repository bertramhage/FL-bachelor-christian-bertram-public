# MIMIC-III Clinical Data Module

This module provides functionality for working with the MIMIC-III (Medical Information Mart for Intensive Care III) clinical database.

## Components

### Core Files
- **main.py**: Entry point for data loading and preprocessing
- **nets.py**: Neural network model architecture for prediction tasks
- **vae_model.py**: Variational Autoencoder (VAE) model for synthetic data generation
- **generate_syn_data.py**: Functions to generate label-conditional synthetic clinical data using the VAE

### Utils
The utils are copied from [YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks)
- **common_utils.py**: Common utility functions shared across the module
- **feature_extractor.py**: Functions for extracting features from clinical time series data
- **metrics.py**: Custom evaluation metrics for model assessment
- **preprocessing.py**: Data preprocessing and transformation utilities
- **readers.py**: Functions for reading and parsing MIMIC-III data formats
- **utils.py**: Additional helper utilities

### Resources
- **channel_info.json**: One-hot-encodings for categorical features
- **discretizer_config.json**: Configs for the discretizer