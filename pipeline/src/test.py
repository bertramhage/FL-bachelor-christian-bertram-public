import torch
import numpy as np
from torch import nn
from argparse import Namespace
from torch.utils.data import DataLoader
from pandas import DataFrame
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from pipeline.src.helpers import get_loss

def eval_model(net: nn.Module,
               val_loader: DataLoader | list[tuple[DataLoader, DataLoader]],
               args) -> tuple[float, float, float]:
    """
    Evaluate the model on the validation set.
    
    Args:
        net (nn.Module): The model to evaluate.
        val_loader (DataLoader | list[tuple[DataLoader, DataLoader]]): The validation data loader or a list of tuples of (train_loader, val_loader).
        args: Additional arguments.
    
    Returns:
        tuple: A tuple containing accuracy, loss, and AUC score.
    """
    
    if args.no_fl:
        return test_model(net, val_loader, args=args)
    else:
        if not isinstance(val_loader[0], tuple):
            raise ValueError("Early stopping without test set requires validation loaders, but they were not provided.")
        else:
            client_loaders = val_loader
        
        all_val_acc, all_val_loss, all_val_auc = [], [], []
        num_val_samples = 0
        total_weighted_acc, total_weighted_loss, total_weighted_auc = 0, 0, 0

        for _, val_loader in client_loaders: # Iterate through (train_loader, val_loader) tuples
            client_val_acc, client_val_loss, client_val_auc = test_model(net, val_loader, args=args)
            num_samples = len(val_loader.dataset)
            
            all_val_acc.append(client_val_acc)
            all_val_loss.append(client_val_loss)
            all_val_auc.append(client_val_auc)
            
            # Weighted average calculation
            total_weighted_acc += client_val_acc * num_samples
            total_weighted_loss += client_val_loss * num_samples
            if np.isnan(client_val_auc) and args.early_stopping_metric == 'auc':
                num_samples = 0
            elif np.isnan(client_val_auc):
                pass
            else:
                total_weighted_auc += client_val_auc * num_samples
            num_val_samples += num_samples

        # Calculate weighted average metrics
        acc_eval = total_weighted_acc / num_val_samples if num_val_samples > 0 else 0
        loss_eval = total_weighted_loss / num_val_samples if num_val_samples > 0 else float('inf')
        auc_eval = total_weighted_auc / num_val_samples if num_val_samples > 0 else 0
        return acc_eval, loss_eval, auc_eval

def test_model(net, test_loader: DataLoader, args: Namespace) -> tuple[float, float, float]:
    """
    Evaluate the model on the test set.
    Args:
        net (nn.Module): The model to evaluate.
        test_loader (DataLoader): The test data loader.
        args: Additional arguments, including:
            - device (torch.device): Device to run the model on
            - loss (str): Loss function to use
            - threshold (float): Threshold for classification.
    Returns:
        tuple: A tuple containing accuracy, average loss, and AUC score.
    """
    net.eval()  # Set model to evaluation mode
    correct, total, loss_total = 0, 0, 0
    loss_func = get_loss(args.loss)
    threshold = args.threshold

    all_labels = []
    all_outputs = []
    with torch.no_grad():  # Disable gradient computation
        for features, labels in test_loader:
            features, labels = features.to(args.device), labels.to(args.device)
            if isinstance(loss_func, torch.nn.BCEWithLogitsLoss): 
                labels = labels.unsqueeze(1).float()
            outputs = net(features)
            loss = loss_func(outputs, labels)
            loss_total += loss.item()

            if isinstance(loss_func, torch.nn.BCEWithLogitsLoss):
                # Apply sigmoid to get probabilities, then threshold at 0.5
                probs = torch.sigmoid(outputs)
                predicted = (probs > threshold).float()
            else:
                _, predicted = torch.max(outputs, 1)
                
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Save outputs and labels (move to CPU for further processing)
            all_labels.append(labels.cpu())
            all_outputs.append(outputs.cpu())

    accuracy = correct / total
    avg_loss = loss_total / len(test_loader)
    
    all_labels = torch.cat(all_labels)
    all_outputs = torch.cat(all_outputs)

    if isinstance(loss_func, torch.nn.BCEWithLogitsLoss):
        # For binary classification, use sigmoid and take the only column (index 0)
        probs = torch.sigmoid(all_outputs).numpy()   # shape: [N, 1]
        auc = roc_auc_score(all_labels.numpy(), probs[:, 0])
    else:
        probs = torch.softmax(all_outputs, dim=1).numpy()
        auc = roc_auc_score(all_labels.numpy(), probs[:, 1])
    
    return accuracy, avg_loss, auc

def calculate_test_scores(net, test_loader: DataLoader, args: Namespace) -> DataFrame:
    """
    Calculate final test scores on the test set.

    Args:
        net (nn.Module): The model to evaluate.
        test_loader (DataLoader): The test data loader.
        args: Additional arguments, including:
            - device (torch.device): Device to run the model on
            - loss (str): Loss function to use
            - threshold (float): Threshold for classification.
    
    Returns:
        pd.DataFrame: A DataFrame containing accuracy, AUC, precision, recall, F1-score, and Average Precision.
    """
    net.eval()
    all_labels = []
    all_outputs = []
    all_preds = []
    threshold = args.threshold
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(args.device), labels.to(args.device)
            if args.loss == 'bce_with_logits': 
                labels = labels.unsqueeze(1).float()
            outputs = net(features)
            
            all_labels.append(labels.cpu())
            all_outputs.append(outputs.cpu())
            if args.loss == 'bce_with_logits':
                probs = torch.sigmoid(outputs)
                predicted = (probs > threshold).float()
            else:
                _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu())
    
    # Concatenate all batches
    all_labels = torch.cat(all_labels).numpy()
    all_outputs = torch.cat(all_outputs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    # Compute classification metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    if args.loss == 'bce_with_logits':
        probs = torch.sigmoid(torch.tensor(all_outputs)).numpy()
        ap = average_precision_score(all_labels, probs[:, 0])
        auc = roc_auc_score(all_labels, probs[:, 0])
    else:
        probs = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
        ap = average_precision_score(all_labels, probs[:, 1])
        auc = roc_auc_score(all_labels, probs[:, 1])
    
    import pandas as pd
    data = {
        'Accuracy': [accuracy],
        'AUC': [auc],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1],
        'AP': [ap]
    }
    return pd.DataFrame(data)