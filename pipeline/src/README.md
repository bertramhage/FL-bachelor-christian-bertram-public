# Pipeline Source Code

This directory contains the core code implementation of the Federated Learning (FL) pipeline components.

## Core Modules

### Federated Learning Core
- **strategy.py**: Implements federated aggregation strategies: FedAvg (Federated Averaging) for model weights.
- **update.py**: Contains model update procedures for both federated and centralized training, including early stopping mechanisms.

### Testing and logging
- **test.py**: Implements model evaluation functions with various metrics (accuracy, precision, recall, F1, AUC).
- **statistics.py**: Contains functions for calculating and logging data statistics, including label distributions.
- **logging.py**: Custom logging functionality for experiment tracking and result management.

### Utility Modules
- **non_iid_partitioning.py**: Provides functions for creating non-IID (non-Independent and Identically Distributed) data partitions for simulating heterogeneous client data in federated learning scenarios. Includes beta distribution-based partitioning.
- **helpers.py**: General utility functions for data loading, model initialization, and other common operations.


## Dataset-Specific Modules

### Brain Tumor Module (`braintumor/`)
Implementation for working with the Brain Tumor MRI image dataset.\
See [README for Brain Tumor Module](braintumor/README.md)

### MIMIC-III Module (`mimiciii/`)
Implementation for working with the MIMIC-III clinical dataset.\
See [README for MIMIC-III Module](mimiciii/README.md)