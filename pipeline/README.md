# Federated Learning Pipeline

This directory contains the implementation of a machine learning pipeline that supports both centralized (CL) and federated learning (FL) frameworks.

## Structure

- `main.py`: Main pipeline file. Orchestrates the entire workflow including data loading, model training, and evaluation.
- `arguments.py`: Defines all command-line arguments for configuring experiments (dataset choice, FL parameters, model hyperparameters, etc.).
- `src/`: Contains the core implementation of the pipeline components. See the [src README](./src/README.md) for details.

## Prefect Integration

This project uses Prefect for workflow orchestration. The pipeline is defined as a Prefect flow, which provides:

1. **Visualization**: Track experiments through the Prefect UI
2. **Monitoring**: Monitor runs
3. **Logging**: Comprehensive logging of execution details
4. **Cloud Result Storage**: Through Prefect artifacts which can be viewed in the UI or retrieved using the Prefect [Python SDK](https://github.com/PrefectHQ/prefect).

To use Prefect features, you need to set up Prefect environment variables:
```bash
export PREFECT_API_KEY=your_api_key
export PREFECT_API_URL=your_api_url
```

## Basic Usage

Run the pipeline with a specific dataset and parameters:

```bash
python -m pipeline.main --dataset braintumor --rounds 100 --num_clients 8
```

Add tags for better organization in Prefect:

```bash
python -m pipeline.main --tags="experiment,braintumor" --dataset=braintumor
```

## Common Parameters

- `--dataset`: Choose which dataset to use (`braintumor`, or `mimiciii`)
- `--no_fl`: Use centralized training instead of federated learning
- `--num_clients`: Number of clients for federated learning
- `--rounds`: Number of communication rounds
- `--non_iid_alpha`: Control the non-IID distribution of data (lower values = more heterogeneous)
- `--samples_per_client`: Number of samples per client
- `--early_stopping`: Enable early stopping
- `--run_name`: Name for the experiment run

For a complete list of available parameters, refer to `arguments.py` or run:

```bash
python -m pipeline.main --help
```

## Current used parameters
In the project we use the following settings for each dataset, respectively:

### Braintumor dataset
```bash
python -um pipeline.main \
    --dataset braintumor \
    --rounds 200 \
    --num_clients 8 \
    --samples_per_client 25 \
    --lr 0.001 \
    --dropout 0.3 \
    --batch_size 10 \
    --local_ep 10 \
    --weight_decay 1e-4 \
    --gradient_clipping
```

### MIMIC-III 
```bash
python -um pipeline.main \
    --dataset mimiciii \
    --rounds 200 \
    --num_clients 8 \
    --samples_per_client 2000 \
    --remove_outliers \
    --early_stopping \
    --lr 0.0005 \
    --dropout 0.3 \
    --batch_size 8  \
    --local_ep 5
```

## Results
Results are both stored locally using the Tensorboard `SummaryWriter`, as well as saved to Prefect Cloud as Prefect Artifacts.

### Tensorboard
For accessing the results in Tensorboard specify a `result_dir` path which will save the results to `pipeline/runs/{result_dir}`.\
Run `tensorboard --logdir=pipeline/runs/{result_dir}` to open Tensorboard in the browser.

### Prefect Arifacts
Prefect artifacts are accesed in the Prefect Cloud UI or via the [REST API](https://docs.prefect.io/v3/api-ref/rest-api) or [Prefect Python SDK](https://github.com/PrefectHQ/prefect).\
Use the SDK for easy code access for further evaluation (add neccesary filters):
```python
from prefect.client.orchestration import get_client

async with get_client() as client:
    result = await client.read_flow_runs()
```

The following artifacts are stored:
**Step-wise test metrics**:
- `pipeline-data-train-accuracy`: Training accuracy metrics
- `pipeline-data-train-auc`: Training AUC-ROC scores
- `pipeline-data-train-loss`: Training loss values
- `pipeline-data-test-accuracy`: Test accuracy metrics
- `pipeline-data-test-loss`: Test loss values
- `pipeline-data-test-auc`: Test AUC-ROC scores

**Run-wise evaluations**:
- `pipeline-data-label-distribution`: Distribution of labels across clients
- `pipeline-data-test-scores`: Final test metrics

**Run report**:
- `pipeline-report`: Comprehensive markdown report with embedded visualizations
