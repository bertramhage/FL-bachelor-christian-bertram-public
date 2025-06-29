# Cross-Silo Federated Learning: Advancing Robustness and Scalability for Practical Applications Repo

A bachelor project focused on implementing and evaluating Federated Learning (FL) techniques for healthcare datasets.

## Repository Overview

This repository contains the implementation of a machine learning pipeline that supports both centralized and federated learning approaches. The code is organized as follows:

- **pipeline/**: Contains the core ML pipeline implementation
  - `main.py`: Main entry point for running experiments
  - `arguments.py`: Command-line argument definitions
  - `src/`: Source code for model implementations and dataset-specific helpers
- **data/**: Storage location for datasets
  - `brain_tumor_dataset/`: Brain tumor image dataset (binary classification)
  - `mimiciii/`: MIMIC-III clinical dataset files
- **preprocessing/**: Preprocessing of the raw data
  - `mimic3benchmark`: Altered clone of preprocessing from [mimic3-benchmark](https://github.com/YerevaNN/mimic3-benchmarks/tree/master/mimic3benchmark)
- **evaluation/**: Tools for evaluating model performance
  - `helpers.py`: Helper functions for evaluation
  - `main.py`: Main evaluation functions
  - `plot_examples.py`: Visualization utilities
  - `evaluations/`: Jupyter notebooks with evaluation results
- **experiments/**: Jupyter notebooks with experimental results
  - Various experiments for FL with different datasets and settings
- **paper/**: LaTeX source for the research paper
  - `main.tex`: Main paper file
  - `Sections/`: Paper sections
  - `references.bib`: Bibliography
- **Litterature/**: Collection of research papers related to federated learning

## Datasets

This project uses two main datasets:

1. **Brain Tumor Dataset**: A dataset of brain MRI images labeled as either containing tumors (yes) or not (no).

2. **MIMIC-III**: A large clinical dataset containing de-identified health data associated with hospital stays. We use the in-hospital mortality prediction task.

**Note**: The data folder is not included in the GitHub repository due to size constraints and privacy concerns. To use this code, you will need to obtain these datasets separately.

## Python Environment Setup

To run this code, you need Python 3.11.9 and the following packages:

```bash
# Create a new virtual environment
python -m venv fl-env
source fl-env/bin/activate  # On Windows: fl-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The main dependencies include:
- torch (2.6.0)
- numpy (1.26.4)
- tensorflow (2.18.0)
- scikit-learn (1.6.1)
- prefect (3.2.15)
- pandas (2.2.3)
- and others listed in requirements.txt

## Running the Pipeline

The ML pipeline can be executed using the `pipeline/main.py` script with various command-line arguments defined in `pipeline/arguments.py`.

### Basic Usage

```bash
python -um pipeline.main --dataset braintumor --rounds 100 --num_clients 8
```

### Common Parameters

- `--dataset`: Choose which dataset to use (`braintumor`, or `mimiciii`)
- `--no_fl`: Use centralized training instead of federated learning
- `--num_clients`: Number of clients for federated learning
- `--rounds`: Number of communication rounds
- `--non_iid_alpha`: Control the non-IID distribution of data (lower values = more heterogeneous)
- `--samples_per_client`: Number of samples per client
- `--early_stopping`: Enable early stopping
- `--run_name`: Name for the experiment run

### Full list of parameters

For a complete list of available parameters, refer to the `pipeline/arguments.py` file or run:

```bash
python -m pipeline.main --help
```

### Current used parameters
The following is the basic settings for running with each of the datasets

#### Braintumor dataset
```bash
python -um pipeline.main \
    --dataset braintumor \
    --rounds 500 \
    --num_clients 8 \
    --lr 0.001 \
    --samples_per_client 25 \
    --non_iid_alpha inf
```

#### MIMIC-III 
```bash
python -um pipeline.main \
    --dataset mimiciii \
    --rounds 200 \
    --num_clients 8 \
    --samples_per_client 800 \
    --non_iid_alpha inf \
    --remove_outliers \
    --early_stopping \
    --lr 0.0005 \
    --batch_size 8 
```

## Prefect Integration

This project uses Prefect for workflow orchestration. The pipeline is defined as a Prefect flow, which provides:

1. **Visualization**: Track experiments through the Prefect UI
2. **Monitoring**: Monitor runs
4. **Logging**: Comprehensive logging of execution details

To use Prefect features, you need to:

1. Set up Prefect environment variables:
   ```bash
   export PREFECT_API_KEY=your_api_key
   export PREFECT_API_URL=your_api_url
   ```

2. Run with tags for organization:
   ```bash
   python -m pipeline.main --tags="experiment,braintumor" --dataset=braintumor
   ```

## Paper

The findings of this research are documented in the paper located in the `paper/` directory. The compiled PDF is available at `paper/main.pdf`.