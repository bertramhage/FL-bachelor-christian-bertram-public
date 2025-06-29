import os
import polars as pl
import numpy as np
from joblib import Memory
from argparse import Namespace
from pipeline.src.mimiciii.utils.preprocessing import Discretizer
from torch import Tensor
from typing import Literal
from functools import lru_cache
import json

def get_data(args: Namespace) -> tuple[np.ndarray, np.ndarray, Namespace]:
    """
    Load and preprocess MIMIC-III data.

    Args:
        args (Namespace): Arguments:
            - mimiciii_path (str): Path to the MIMIC-III dataset.
            - gcp_credentials (str): Path to GCP credentials file.
            - outlier_strategy (str | list[str]): Strategy for outlier removal.
    
    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): Preprocessed feature data.
            - y (np.ndarray): Labels.
            - args (Namespace): Updated arguments with additional attributes:
                - features (tuple): Start and end dimensions of features.
                - n_features (int): Number of features.
                - n_classes (int): Number of classes.
    """
    X, y = _get_mimic_data(args.mimiciii_path, args.gcp_credentials, args.outlier_strategy)
    args.features = (1,2) # start dim, end dim
    args.n_features = 76
    args.n_classes = 2
    return X, y, args

# Cache the function to avoid continuous loading from the remote storage
memory = Memory(os.path.join(os.path.dirname(__file__), '__cache__'), verbose=0)
@memory.cache
def _get_mimic_data(mimiciii_path,
                    gcp_credentials,
                    outlier_removal_strategy
                    ) -> tuple[np.ndarray, np.ndarray, Namespace]:

    X, y = load_raw_data(mimiciii_path=mimiciii_path,
                         gcp_credentials=gcp_credentials,
                         outlier_removal_strategy=outlier_removal_strategy)
    X, y = descretize_data(X, y)
    
    return X, y

def load_raw_data(mimiciii_path: str,
                  gcp_credentials: str,
                  N: int = None,
                  outlier_removal_strategy: str | list[str] | None = None) -> tuple[list[np.ndarray], np.ndarray, Namespace]:
    """ Load and preprocess raw MIMIC-III data from the remote storage.
    
    Args:
        mimiciii_path (str): Path to the MIMIC-III dataset.
        gcp_credentials (str): Path to GCP credentials file.
        N (int, optional): Number of rows to load. If None, loads all rows.
        outlier_removal_strategy (str | list[str] | None): Strategy for outlier removal.
    
    Returns:
        tuple: A tuple containing:
            - X (list[np.ndarray]): Preprocessed feature data.
            - y (np.ndarray): Labels.
    """
    global columns
    
    if 'gs://' in mimiciii_path:
        storage_opt = {"SERVICE_ACCOUNT": gcp_credentials}
        data = pl.scan_parquet(mimiciii_path, storage_options=storage_opt)
    else:
        data = pl.scan_parquet(mimiciii_path)
        
    if N:
        data = data.head(N).collect()
    else:
        data = data.collect()
        
    if outlier_removal_strategy is not None:
        data, removed_rows = remove_outliers(data, strategy=outlier_removal_strategy)
        print(f"Removed {removed_rows} rows due to outlier strategy")
        
    data = data.fill_nan('').fill_null('') # Discretizer expects null values to be empty strings
    
    target = data.unique(subset=['ICUSTAY_ID', 'HADM_ID'], maintain_order=True).get_column('mortality')
    partitioned_data = data.partition_by('ICUSTAY_ID', 'HADM_ID')
    X = [df.select(columns).to_numpy() for df in partitioned_data]
    y = target.to_numpy()
        
    return X, y

def descretize_data(X: list[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Discretize the raw data using the Discretizer class (see pipeline/src/mimiciii/utils/preprocessing.py). """
    data = X
    ts = [48.0] * len(data)
    labels = y
    
    discretizer = _get_discretizer()
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    whole_data = (np.array(data), labels)
    
    return whole_data

def _get_discretizer() -> Discretizer:
    return Discretizer(timestep=1.0,
                       store_masks=True,
                       impute_strategy='previous',
                       start_time='zero')

@lru_cache(maxsize=2)
def _get_discretizer_header(args):
    X, _ = load_raw_data(mimiciii_path=args.mimiciii_path, gcp_credentials=args.gcp_credentials, N=1)
    return _get_discretizer().transform(X[0])[1].split(',')

def _get_cont_channels(args: Namespace) -> list:
    discretizer_header = _get_discretizer_header(args)
    return [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

def normalize_data(X_tensor: Tensor, args: Namespace) -> Tensor:
    """ Normalize continuous channels in the data tensor using z-score standardization. """
    cont_channels = _get_cont_channels(args)
    for c in cont_channels:
        mean = X_tensor.flatten(0,1)[:,c].mean()
        std = X_tensor.flatten(0,1)[:,c].std()
        X_tensor[:,:,c] = (X_tensor[:,:,c] - mean) / std
    
    return X_tensor

def remove_outliers(df: pl.DataFrame, strategy: str | list[str]) -> tuple[pl.DataFrame, int]:
    """
    Remove outliers from the DataFrame based on the specified strategy.
    Args:
        df (pl.DataFrame): Input DataFrame containing continuous channels.
        strategy (str | list[str]): Strategy for outlier removal. Options are 'zscore', 'iqr', 'mad'.
    
    Returns:
        tuple: A tuple containing:
            - df (pl.DataFrame): DataFrame with outliers removed.
            - removed_rows (int): Number of rows removed due to outliers.
    """
    rows_before_removal = df.height
    
    config_path='pipeline/src/mimiciii/resources/discretizer_config.json'
    with open(config_path) as f:
        config = json.load(f)
        cont_channels = []
        for key, value in config['is_categorical_channel'].items():
            if value == False:
                cont_channels.append(key)

    # Convert the provided strategy parameter to a set for fast membership testing
    if isinstance(strategy, str):
        strategies = {strategy}
    else:
        strategies = set(strategy)

    # For each channel in cont_channels, compute statistics and mark outliers
    updated_exprs = []
    for c in cont_channels:
        # Collect the column values as a Series for computing statistics
        col_series = df.select(pl.col(c))[c]
        conds = []
        
        # Z-score method: mark as outlier if |(x - mean) / std| > 3
        if 'zscore' in strategies:
            mean_val = col_series.mean()
            std_val = col_series.std()
            if std_val == 0:
                zcond = pl.lit(False)
            else:
                zcond = ((pl.col(c) - mean_val).abs() / std_val) > 3.0
            conds.append(zcond)
        
        # IQR method: mark as outlier if x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR
        if 'iqr' in strategies:
            q1 = col_series.quantile(0.25)
            q3 = col_series.quantile(0.75)
            iqr = q3 - q1
            iqr_cond = (pl.col(c) < q1 - 1.5 * iqr) | (pl.col(c) > q3 + 1.5 * iqr)
            conds.append(iqr_cond)
        
        # MAD method: mark as outlier if |x - median| > 3 * MAD
        if 'mad' in strategies:
            median_val = col_series.median()
            mad = float(np.median(np.abs(col_series.to_numpy() - median_val)))
            mad_cond = (pl.col(c) - median_val).abs() > (3.0 * mad)
            conds.append(mad_cond)
        
        # Combine all outlier conditions using OR. If any condition is met, treat the value as an outlier.
        if conds:
            combined_cond = conds[0]
            for cond in conds[1:]:
                combined_cond = combined_cond | cond
            updated_expr = pl.when(combined_cond).then(None).otherwise(pl.col(c)).alias(c)
            updated_exprs.append(updated_expr)
        else:
            # No strategies provided for this channel; leave it unchanged
            updated_exprs.append(pl.col(c))
    
    # Update the dataframe: only modify the columns in cont_channels based on the computed expressions
    df = df.with_columns(updated_exprs)
    
    # Remove rows where all values in the selected continuous channels are null
    import functools, operator
    filter_condition = functools.reduce(operator.or_, [pl.col(c).is_not_null() for c in cont_channels])
    df = df.filter(filter_condition)
    
    # Calculate the number of rows removed
    rows_after_removal = df.height
    removed_rows = rows_before_removal - rows_after_removal
    
    return df, removed_rows

columns = ['Hours',
           'Capillary refill rate',
           'Diastolic blood pressure',
           'Fraction inspired oxygen',
           'Glascow coma scale eye opening',
           'Glascow coma scale motor response',
           'Glascow coma scale total',
           'Glascow coma scale verbal response',
           'Glucose',
           'Heart Rate',
           'Height',
           'Mean blood pressure',
           'Oxygen saturation',
           'Respiratory rate',
           'Systolic blood pressure',
           'Temperature',
           'Weight',
           'pH']