import torch
import os
import argparse
import concurrent.futures
import pandas as pd
import asyncio
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import copy
import numpy as np
import random

from pipeline.arguments import add_arguments
from pipeline.src.braintumor.nets import CNN
from pipeline.src.mimiciii.nets import LSTM
from pipeline.src.helpers import (get_dataloader,
                                  balancing,
                                  balance_synthetic_data,
                                  get_prefect_flow_name,
                                  log_results)
from pipeline.src.logging import ResultLogger
from pipeline.src.update import ModelUpdateFL, ModelUpdateCentralized, EarlyStopping
from pipeline.src.statistics import calculate_label_distribution, log_label_distribution_to_tensorboard
from pipeline.src.strategy import FedAvg
from pipeline.src.test import test_model, calculate_test_scores, eval_model
from pipeline.src.non_iid_partitioning import create_beta_partitions
from tqdm import trange, tqdm
from importlib import import_module
from prefect import flow, task, tags
from prefect.artifacts import create_progress_artifact, update_progress_artifact
from prefect.context import get_run_context
from typing import Literal


@task
def train_network(args,
                  net_glob: torch.nn.Module,
                  train_loader: torch.utils.data.DataLoader,
                  test_loader: torch.utils.data.DataLoader,
                  X_train_tensor: torch.Tensor,
                  y_train_tensor: torch.Tensor,
                  client_loaders: list[torch.utils.data.DataLoader] | list[tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]):
    
    try: # Get the current Prefect flow run name
        context = get_run_context().task_run
        prefect_run_name = asyncio.run(get_prefect_flow_name(context)) + ', '
    except:
        prefect_run_name = '' # Fallback to empty string

    run_name = "/" + args.run_name if args.run_name != "" else ""
    
    # Set up TensorBoard writer
    os.makedirs(f'pipeline/runs/{args.result_dir}', exist_ok=True)
    writer = SummaryWriter(f"pipeline/runs/{args.result_dir}{run_name}, \
    {prefect_run_name}\
    {'Centralized' if args.no_fl else 'Federated'}, \
    {args.rounds} rounds, \
    {args.samples_per_client} samples per client, \
    {args.synthetic_count} synthetic samples, \
    {args.non_iid_alpha} alpha, {__import__('datetime').datetime.now()}")
        
    
    logger = ResultLogger() # Prefect logger
    
    if args.early_stopping:
        early_stopping = EarlyStopping(net_glob, args)
        print(f"Early stopping initialized with patience: {early_stopping.patience}, metric: {early_stopping.metric}")
    
    if args.no_fl: # Centralized training
        
        if args.early_stopping and not args.early_stopping_use_testset:
            # Create validation set
            X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
                X_train_tensor, y_train_tensor, test_size=args.test_size, random_state=args.seed, shuffle=True
            )
            train_loader = get_dataloader(X_train_tensor, y_train_tensor, args.batch_size)
            val_loader = get_dataloader(X_val_tensor, y_val_tensor, args.batch_size)
            print(f"Created validation set. New train shape: {X_train_tensor.shape}, Val shape: {X_val_tensor.shape}")
        
        if args.balancing is not None:
            print(f"Balancing dataset using {args.balancing} method")
            print(f"Label distribution before balancing: {calculate_label_distribution([train_loader])}")
            X_train_tensor, y_train_tensor = balancing(X_train_tensor, y_train_tensor, args.balancing, args.ir, args.seed)
            train_loader = get_dataloader(X_train_tensor, y_train_tensor, args.batch_size)
            print(f"Label distribution after balancing: {calculate_label_distribution([train_loader])}.\
                Total samples after balancing: {X_train_tensor.shape[0]}")

        updater = ModelUpdateCentralized(args, net_glob, train_loader)
        
    else: # Federated training
        
        has_val_loaders = isinstance(client_loaders[0], tuple) if client_loaders else False
        
        # Log label distribution
        client_train_loaders = [loader[0] if has_val_loaders else loader for loader in client_loaders]
        log_label_distribution_to_tensorboard(writer, client_train_loaders, global_step=0)
        logger.add_table("label-distribution", pd.DataFrame(calculate_label_distribution(client_train_loaders)))
        
        def _train_client(loader_input):
            # Use only the train loader if validation loaders are present
            client_loader = loader_input[0] if has_val_loaders else loader_input
            
            # Copy global model for client update
            local_model = copy.deepcopy(net_glob)
            local_update = ModelUpdateFL(args, local_model, client_loader)
            local_weights, _ = local_update()  # Train local model
            return local_weights
    
    progress_artifact_id = create_progress_artifact( # Prefect progress artifact
        progress=0.0,
        description="Indicates the training progress",
    )
    i = 0
    for round in trange(args.rounds,
                        desc="Round" if not args.no_fl else "Epoch"
                        ):
        # Train the model for one epoch / round
        if args.no_fl:
            _ = updater()
        else:
            local_models = []
            train_loaders_for_executor = client_loaders
            
            # Run in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                local_models = list(executor.map(_train_client, train_loaders_for_executor))
            
            # Aggregate models using FedAvg
            w_glob = FedAvg(local_models, args)
            net_glob.load_state_dict(w_glob)
            
        # Test model
        acc_train, loss_train, auc_train = test_model(net_glob, train_loader, args=args)
        acc_test, loss_test, auc_test = test_model(net_glob, test_loader, args=args)
        log_results(writer, logger,
                    acc_train, loss_train, auc_train,
                    acc_test, loss_test, auc_test,
                    round)
        
        # Early stopping check
        if args.early_stopping:
            if args.early_stopping_use_testset:
                acc_eval, loss_eval, auc_eval = test_model(net_glob, test_loader, args=args)
            else:
                acc_eval, loss_eval, auc_eval = eval_model(net_glob,
                                                           client_loaders if not args.no_fl else val_loader,
                                                           args=args)
            
            if early_stopping(net_glob, acc_eval, loss_eval, auc_eval, round):
                print(f"Early stopping triggered at step {round}.")
                net_glob.load_state_dict(early_stopping.get_best_model()) # Restore best model weights
                break
        else:
            acc_eval, loss_eval, auc_eval = acc_test, loss_test, auc_test
        
        print(f"{'Epoch' if args.no_fl else 'Round'} \
            {round}: Eval Loss: {loss_eval:.4f}, \
                Eval Accuracy: {acc_eval:.4f}, \
                    Eval AUC-ROC: {auc_eval:.4f}")
        
        i += 1
        update_progress_artifact(artifact_id=progress_artifact_id, progress=(i/args.rounds)*100)

    print("Calculating test scores")
    test_scores = calculate_test_scores(net_glob, test_loader, args)
    logger.add_table("test-scores", test_scores)
    writer.add_text("Test Scores", str(test_scores), 0)
    
    writer.close()
    logger.save_raw_data('pipeline-data')
    logger.create_report()

@task
def create_clients(args,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   normalizer: callable):
    if args.samples_per_client is not None:
        args.client_sizes = np.ones(args.num_clients, dtype=int) * args.samples_per_client
    else:
        args.client_sizes = np.ones(args.num_clients, dtype=int) * int(np.floor(X_train.shape[0] / args.num_clients))

    client_partitions = create_beta_partitions(X_train, y_train,
                            num_partitions=args.num_clients,
                            samples_per_partition=args.client_sizes,
                            alpha=args.non_iid_alpha  # or any alpha you want
                            )
    client_data, syn_list_x, syn_list_y = [], [], []

    def process_client_partition(client_partition):
        client_x, client_y = client_partition
        client_x_tensor = normalizer(torch.tensor(client_x, dtype=torch.float32), args)
        client_y_tensor = torch.tensor(client_y, dtype=torch.long)
        
        client_syn_x, client_syn_y = None, None
        if args.synthetic_count > 0:
            synthetic_module = getattr(import_module(f'pipeline.src.{args.dataset}.generate_syn_data'), 'train_and_generate_by_label')
            client_syn_x, client_syn_y = synthetic_module(client_x_tensor, client_y_tensor, **vars(args))
        
        if args.balancing is not None:
            client_x_tensor, client_y_tensor = balancing(client_x_tensor, client_y_tensor, args.balancing, args.ir, args.seed)
        
        return (client_x_tensor, client_y_tensor), client_syn_x, client_syn_y

    print(f"Processing {len(client_partitions)} client partitions in parallel...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_client_partition, client_partitions),
            total=len(client_partitions),
            desc='Generating client datasets'
        ))
    
    for client_data_item, client_syn_x, client_syn_y in results:
        client_data.append(client_data_item)
        if client_syn_x is not None and client_syn_y is not None:
            syn_list_x.append(client_syn_x)
            syn_list_y.append(client_syn_y)
        
    if args.synthetic_count>0:
        # Merge synthetic samples across clients and shuffle them
        all_syn_x = torch.cat(syn_list_x, dim=0)
        all_syn_y = torch.cat(syn_list_y, dim=0)
        perm = torch.randperm(all_syn_x.size(0))
        all_syn_x, all_syn_y = all_syn_x[perm], all_syn_y[perm]
        if all_syn_x.size(0) < args.synthetic_count * args.num_clients:
            raise ValueError("Not enough synthetic samples available.")

        # Append the appropriate synthetic samples to each client's original data and create DataLoader
        enriched_data = []
        for i, (client_x_tensor, client_y_tensor) in enumerate(client_data):
            if not args.synthetic_balancing:
                start, end = i * args.synthetic_count, (i + 1) * args.synthetic_count
                client_syn_x, client_syn_y = all_syn_x[start:end], all_syn_y[start:end]
                combined_x = torch.cat([client_x_tensor, client_syn_x], dim=0)
                combined_y = torch.cat([client_y_tensor, client_syn_y], dim=0)
                
                perm_client = torch.randperm(combined_x.size(0))
                enriched_data.append((combined_x[perm_client], combined_y[perm_client]))
            else:
                enriched_data.append(balance_synthetic_data(client_x_tensor, client_y_tensor, all_syn_x, all_syn_y))
        client_data = enriched_data
    
    # If early stopping -> create validation set
    if args.early_stopping and not args.early_stopping_use_testset:
        client_train_data, client_val_data = [], []
        for _, (client_x_tensor, client_y_tensor) in enumerate(client_data):
            client_train_x_tensor, client_val_x_tensor, client_train_y_tensor, client_val_y_tensor = train_test_split(
                client_x_tensor, client_y_tensor, test_size=args.test_size, random_state=args.seed, shuffle=True
            )
            assert client_train_x_tensor.shape[0] > 0, "Client train data is empty. Use more samples per client or reduce test size."
            assert client_val_x_tensor.shape[0] > 0, "Client val data is empty. Use more samples per client or increase test size."
            
            client_train_data.append((client_train_x_tensor, client_train_y_tensor))
            client_val_data.append((client_val_x_tensor, client_val_y_tensor))
        
        client_loaders = [(
            get_dataloader(client_train_x_tensor, client_train_y_tensor, args.batch_size),
            get_dataloader(client_val_x_tensor, client_val_y_tensor, args.batch_size)
            ) for (client_train_x_tensor, client_train_y_tensor), (client_val_x_tensor, client_val_y_tensor) 
                          in zip(client_train_data, client_val_data
        )]
    else:
        client_loaders = [get_dataloader(client_x_tensor, client_y_tensor, args.batch_size) for (client_x_tensor, client_y_tensor) in client_data]            

    return client_loaders

@task
def get_dataset(args):
    X, y, args = getattr(import_module(f'pipeline.src.{args.dataset}.main'), 'get_data')(args)
    X: np.ndarray
    y: np.ndarray
    return X, y, args
    
@flow(log_prints=True)
def main(no_fl: bool = False,
         dataset: Literal['synthetic', 'braintumor', 'mimiciii'] = 'synthetic',
         num_clients: int = 50,
         rounds: int = 10,
         samples_per_client: int | None = None,
         non_iid_alpha: float | Literal['inf'] | None = 'inf',
         test_size: float = 0.166,
         result_dir: str = 'results',
         remove_outliers: bool = False,
         overwrite_cache: bool = False,
         balancing: Literal['over', 'under', 'smote', None] = None,
         ir: float | None = 1.0,
         total_samples: int | None = None,
         dataset_path: str | None = None,
         gcp_credentials: str | None = None,
         local_ep: int = 10,
         lr: float = 0.001,
         momentum: float = 0.5,
         seed: int = 42,
         batch_size: int = 10,
         weight_decay: float = 0,
         gradient_clipping: bool = False,
         early_stopping: bool = False,
         early_stopping_patience: int = 5,
         early_stopping_metric: Literal['loss', 'accuracy', 'auc'] = 'loss',
         early_stopping_use_testset: bool = False,
         early_stopping_disregard_rounds: int = 0,
         dropout: float = 0.3,
         synthetic_count: int = 0,
         img_size: int = 32,
         run_name: str = 'test',
         synthetic_balancing: bool = False,
         scale_syn: bool = False,
         threshold: float = 0.5,
         ):
    """ Main pipeline function. Parameters are copies from the command line arguments found in pipeline/arguments.py. """
    
    # Convert passed prefect arguments to a namespace-like object
    class ParamsParser:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    args = ParamsParser(
        no_fl=no_fl,
        num_clients=num_clients,
        test_size=test_size,
        rounds=rounds,
        non_iid_alpha=non_iid_alpha,
        dataset=dataset,
        scale_syn=scale_syn,
        overwrite_cache=overwrite_cache,
        result_dir=result_dir,
        samples_per_client=samples_per_client,
        balancing=balancing,
        total_samples=total_samples,
        dataset_path=dataset_path,
        gcp_credentials=gcp_credentials,
        local_ep=local_ep,
        lr=lr,
        momentum=momentum,
        seed=seed,
        batch_size=batch_size,
        remove_outliers=remove_outliers,
        weight_decay=weight_decay,
        gradient_clipping=gradient_clipping,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_disregard_rounds=early_stopping_disregard_rounds,
        dropout=dropout,
        synthetic_count=synthetic_count,
        img_size=img_size,
        run_name=run_name,
        synthetic_balancing=synthetic_balancing,
        early_stopping_metric=early_stopping_metric,
        early_stopping_use_testset=early_stopping_use_testset,
        ir=ir,
        threshold=threshold,
    )
    
    print('Starting pipeline...')
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.remove_outliers:
        args.outlier_strategy = ['iqr']
    else:
        args.outlier_strategy = None
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)     
    torch.cuda.manual_seed_all(args.seed)
    
    if args.overwrite_cache:
        os.system(f'rm -rf pipeline/src/{args.dataset}/__cache__')
        print(f'Cache deleted for {args.dataset} dataset')
        
    print('Loading data')
    X, y, args = get_dataset(args)
    
    print(f"Retrieval complete. Total samples loaded: {X.shape[0]}")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, shuffle=True)
    print(f"Train feature shape: {X_train.shape[0]}, Train label shape: {y_train.shape[0]}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    print(f"Number of class 0 in test set: {np.sum(y_test == 0)}")
    print(f"Number of class 1 in test set: {np.sum(y_test == 1)}")

    # Load normalizer
    normalizer = getattr(import_module(f'pipeline.src.{args.dataset}.main'), 'normalize_data')
    X_train_tensor_normalized = normalizer(X_train_tensor, args)
    X_test_tensor_normalized = normalizer(X_test_tensor, args)
    
    if args.total_samples is not None and args.no_fl:
        # For no_fl
        X_train_tensor_normalized = X_train_tensor_normalized[:args.total_samples]
        y_train_tensor = y_train_tensor[:args.total_samples]
    elif not args.no_fl and args.samples_per_client is not None: # For FL, make sure we limit samples for beta partitioning
        total_samples = int(np.ceil(args.num_clients * args.samples_per_client))
        X_train = X_train[:total_samples]
        y_train = y_train[:total_samples]
    
    train_loader = get_dataloader(X_train_tensor_normalized, y_train_tensor, args.batch_size)
    test_loader = get_dataloader(X_test_tensor_normalized, y_test_tensor, args.batch_size)
        
    if args.dataset == 'braintumor':
        net_glob = CNN(args).to(args.device)
        args.optimizer = 'adam'
        args.loss = 'bce_with_logits'
    elif args.dataset == 'mimiciii':
        net_glob = LSTM(args).to(args.device)
        args.optimizer = 'adam'
        args.loss = 'bce_with_logits'
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    if args.non_iid_alpha == 'inf' or args.non_iid_alpha is None:
        args.non_iid_alpha = np.inf
        
    if not args.no_fl:
        client_loaders = create_clients(args,
                                        X_train, # We use the unnormalized data
                                        y_train,
                                        normalizer)
    else:
        client_loaders = None
        
    net_glob.train()

    print('Starting training')
    train_network(args,
                  net_glob,
                  train_loader,
                  test_loader,
                  X_train_tensor_normalized,
                  y_train_tensor,
                  client_loaders)
    
    print("Done.")
    
if __name__ == '__main__':
    
    if not os.getenv('PREFECT_API_KEY') or not os.getenv('PREFECT_API_URL'):
        raise EnvironmentError("PREFECT_API_KEY and PREFECT_API_URL must be set in the environment variables.")
    
    parser = argparse.ArgumentParser(description='Federated Learning Pipeline')
    add_arguments(parser)
    args = parser.parse_args()
    
    if args.tags is not None:
        prefect_tags = args.tags.split(',')
        del args.tags
        with tags(*prefect_tags):
            main(**vars(args))
    else:
        del args.tags
        main(**vars(args))