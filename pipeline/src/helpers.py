import torch
import numpy as np

from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from pipeline.src.logging import ResultLogger

from typing import Literal
from argparse import Namespace
from prefect.client.orchestration import get_client, FlowRunFilter
from prefect.client.schemas.filters import FlowRunFilterId
from prefect.context import TaskRunContext
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

def log_results(writer: SummaryWriter, logger: ResultLogger,
                 acc_train: float, loss_train: float, auc_train: float,
                 acc_test: float, loss_test: float, auc_test: float,
                 step: int):
    """
    Log training and testing results to TensorBoard and ResultLogger.
    """
    # Log results to TensorBoard
    writer.add_scalar("Train Accuracy", acc_train, step)
    writer.add_scalar("Train AUC-ROC", auc_train, step)
    writer.add_scalar("Train Loss", loss_train, step)
    writer.add_scalar("Test Accuracy", acc_test, step)
    writer.add_scalar("Test AUC-ROC", auc_test, step)
    writer.add_scalar("Test Loss", loss_test, step)
    
    # Log results to ResultLogger
    logger.add_scalar("train-accuracy", acc_train, step)
    logger.add_scalar("train-auc", auc_train, step)
    logger.add_scalar("train-loss", loss_train, step)
    logger.add_scalar("test-accuracy", acc_test, step)
    logger.add_scalar("test-loss", loss_test, step)
    logger.add_scalar("test-auc", auc_test, step)

async def get_prefect_flow_name(task_context: TaskRunContext) -> str:
    try:
        async with get_client() as client:
            flow_runs_filter = FlowRunFilter(id=FlowRunFilterId(any_=[task_context.flow_run_id]))
            result = await client.read_flow_runs(
                limit=1,
                flow_run_filter=flow_runs_filter,
                
            )
            return result[0].name
    except:
        return ""

def get_dataloader(X_tensor: Tensor, y_tensor: Tensor, batch_size: int) -> DataLoader:
    """ Return a torch DataLoader from X and y tensors. """
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def balancing(
    X: Tensor,
    y: Tensor,
    method: Literal["over", "under", "smote"],
    ir: float = 1.0,
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """
    Balance a binary-class dataset using imblearn's built-in samplers.

    Args:
        X (torch.Tensor): Feature tensor shaped (N, …).
        y (torch.Tensor): Label tensor shaped (N,).
        method (Literal["over", "under", "smote"]): 
            - "over"  - RandomOverSampler  
            - "under" - RandomUnderSampler  
            - "smote" - SMOTE synthetic oversampling
        ir (float, default=1.0): Desired imbalance ratio majority / minority after resampling.
            Must be ≥ 1. Setting ir = 1 gives perfectly balanced classes.
        seed (int, default=42): Seed for reproducibility.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The resampled feature and label tensors.
    """
    assert method in {"over", "under", "smote"}, \
        "method must be 'over', 'under', or 'smote'"
    assert ir >= 1.0, "Imbalance ratio (ir) must be ≥ 1."

    X_np = X.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    classes, counts = np.unique(y_np, return_counts=True)
    if len(classes) != 2:
        print("Only one label available. No balancing will be performed.")
        return X, y

    minority_cls = classes[np.argmin(counts)]
    majority_cls = classes[np.argmax(counts)]
    minority_cnt = counts[np.argmin(counts)]
    majority_cnt = counts[np.argmax(counts)]

    if method in {"over", "smote"}:
        # Target minority size: majority / ir
        target_min = int(majority_cnt / ir)
        if target_min <= minority_cnt:
            # Already satisfies desired ratio
            return X, y
        sampling_strategy = {minority_cls: target_min}
        sampler = (
            RandomOverSampler(sampling_strategy=sampling_strategy, random_state=seed)
            if method == "over"
            else SMOTE(sampling_strategy=sampling_strategy, random_state=seed)
        )
    else:  # "under"
        # Target majority size: minority * ir
        target_maj = int(minority_cnt * ir)
        if target_maj >= majority_cnt:
            return X, y
        sampling_strategy = {majority_cls: target_maj}
        sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy, random_state=seed
        )

    if method in {"over", "under"}:
        sampler: RandomOverSampler | RandomUnderSampler
        _ = sampler.fit_resample(X_np[:,0,:], y_np)
        indeces = sampler.sample_indices_
        X_res = X_np[indeces]
        y_res = y_np[indeces]
    else:  # smote
        orig_shape = X_np.shape[1:]
        X_flat = X_np.reshape(X_np.shape[0], -1)
        X_res_flat, y_res = sampler.fit_resample(X_flat, y_np)
        X_res = X_res_flat.reshape(-1, *orig_shape)

    device = X.device
    X_balanced = torch.tensor(X_res, dtype=X.dtype, device=device)
    y_balanced = torch.tensor(y_res, dtype=y.dtype, device=device)

    return X_balanced, y_balanced

def balance_synthetic_data(
    client_x_tensor: torch.Tensor,
    client_y_tensor: torch.Tensor,
    all_syn_x: torch.Tensor,
    all_syn_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ Balance the synthetic data for a client based on the client's labels.
    
    Args:
        client_x_tensor (torch.Tensor): Client's feature tensor.
        client_y_tensor (torch.Tensor): Client's label tensor.
        all_syn_x (torch.Tensor): All synthetic feature samples.
        all_syn_y (torch.Tensor): All synthetic label samples.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Balanced feature and label tensors for the client.
    """
    unique_classes, counts = torch.unique(client_y_tensor, return_counts=True)
    class_counts = {int(cls.item()): int(cnt.item()) for cls, cnt in zip(unique_classes, counts)}
    
    for target_class in (0, 1):
        if target_class not in class_counts:
            class_counts[target_class] = 0

    target_count = max(class_counts.values())
    
    synthetic_x_list = []
    synthetic_y_list = []
    
    # For each class (0 and 1), calculate the deficit and add synthetic samples if needed.
    for cls in (0, 1):
        count = class_counts[cls]
        deficit = target_count - count
        if deficit > 0:
            cls_mask = (all_syn_y == cls)
            available_indices = torch.nonzero(cls_mask, as_tuple=False).squeeze()
            if available_indices.dim() == 0:
                available_indices = available_indices.unsqueeze(0)
            
            rand_perm = torch.randperm(available_indices.numel())
            chosen_indices = available_indices[rand_perm][:deficit]
            
            synthetic_x_list.append(all_syn_x[chosen_indices])
            synthetic_y_list.append(all_syn_y[chosen_indices])
    
    if synthetic_x_list:
        client_syn_x = torch.cat(synthetic_x_list, dim=0)
        client_syn_y = torch.cat(synthetic_y_list, dim=0)
    else:
        client_syn_x = torch.empty(0, *client_x_tensor.shape[1:], dtype=client_x_tensor.dtype)
        client_syn_y = torch.empty(0, dtype=client_y_tensor.dtype)
    
    combined_x = torch.cat([client_x_tensor, client_syn_x], dim=0)
    combined_y = torch.cat([client_y_tensor, client_syn_y], dim=0)
    
    perm_client = torch.randperm(combined_x.size(0))
    return combined_x[perm_client], combined_y[perm_client]

def get_loss(loss_type: Literal["cross_entropy", "bce_with_logits"]) -> torch.nn.modules.loss._Loss:
    """ Get the loss function based on string identifier. """
    if loss_type == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    elif loss_type == "bce_with_logits":
        return torch.nn.BCEWithLogitsLoss()

def get_optimizer(optimizer_type: Literal["sgd", "adam"], net: torch.nn.Module, args) -> torch.optim.Optimizer:
    """ Get the optimizer based on string identifier. """
    if optimizer_type == "sgd":
        return torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif optimizer_type == "adam":
        return torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)