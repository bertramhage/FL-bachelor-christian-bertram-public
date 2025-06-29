import torch
import copy
from torch.utils.data import DataLoader
from torch.nn import Module
from argparse import Namespace
from pipeline.src.helpers import get_loss, get_optimizer


class ModelUpdateFL:
    def __init__(self, args: Namespace, net: Module, client_loader: DataLoader):
        """
        Initializes the FL model update.

        Args:
            args: Arguments object containing:
                - local_ep (int): Number of local epochs
                - device (torch.device): Device to run the model on
                - loss (str): Loss function to use
                - optimizer (str): Optimizer to use
                - gradient_clipping (bool): Whether to use gradient clipping
            net: The neural network model to train.
            client_loader: DataLoader for the client's training data.
        """
        self.args = args
        self.net = net
        self.loss_func = get_loss(args.loss)
        self.client_loader = client_loader
        self.gradient_clipping = args.gradient_clipping
        self.optimizer = get_optimizer(args.optimizer, net, args)

    def __call__(self):
        """ Perform local_ep epochs of training on the client's data """
        self.net.train()
        
        optimizer = self.optimizer
        epoch_loss = []

        for _ in range(self.args.local_ep):  # Local training epochs
            batch_loss = []
            for _, (features, labels) in enumerate(self.client_loader):
                features, labels = features.to(self.args.device), labels.to(self.args.device)

                if isinstance(self.loss_func, torch.nn.BCEWithLogitsLoss):
                    labels = labels.unsqueeze(1).float()
                
                # Forward pass
                self.net.zero_grad()
                outputs = self.net(features)
                loss = self.loss_func(outputs, labels)

                # Backward pass
                loss.backward()
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()

                batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return self.net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
class ModelUpdateCentralized:

    def __init__(self, args: Namespace, net: Module, train_loader: DataLoader):
        """
        Initializes the centralized model update.

        Args:
            args: Arguments object containing:
                - device (torch.device): Device to run the model on
                - loss (str): Loss function to use
                - optimizer (str): Optimizer to use
                - gradient_clipping (bool): Whether to use gradient clipping
            net: The neural network model to train.
            train_loader: DataLoader for the training data.
        """
        self.net = net
        self.train_loader = train_loader
        self.criterion = get_loss(args.loss)
        self.optimizer = get_optimizer(args.optimizer, net, args)
        self.device = args.device
        self.gradient_clipping = args.gradient_clipping
        self.max_norm = 1.0

    def __call__(self) -> float:
        """Performs one epoch"""
        self.net.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                target = target.unsqueeze(1).float()

            # Forward + backward + optimize
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = self.criterion(output, target)
            loss.backward()
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_norm)
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)
    
class EarlyStopping:
    def __init__(self, net: Module, args: Namespace):
        """
        Initializes the EarlyStopping class

        Args:
            net (Module): The neural network model to monitor.
            args (Namespace): Arguments containing:
                - early_stopping_metric (str): Metric to monitor for early stopping ('loss', 'accuracy', 'auc').
                - early_stopping_patience (int): Number of rounds to wait before stopping if no improvement.
                - early_stopping_disregard_rounds (int): Number of rounds to disregard before starting to monitor.
        """
        self.metric = args.early_stopping_metric
        self.patience = getattr(args, 'early_stopping_patience', 5)
        self.early_stopping_disregard_rounds = args.early_stopping_disregard_rounds
        self.best_metric = float('-inf') if args.early_stopping_metric in ['accuracy', 'auc'] \
                                else float('inf')
        self.patience_counter = 0
        self.best_model_state = copy.deepcopy(net.state_dict())
    
    def __call__(self, net: Module, acc: float, loss: float, auc: float, step: int) -> bool:
        """
        Checks if early stopping criteria are met.

        Returns True if early stopping should be triggered, otherwise False.
        """
        if not self.early_stopping_disregard_rounds < step:
            return False
        
        current_metric = None
        if self.metric == 'loss':
            current_metric = loss
            is_better = current_metric < self.best_metric
        elif self.metric == 'accuracy':
            current_metric = acc
            is_better = current_metric > self.best_metric
        elif self.metric == 'auc':
            current_metric = auc
            is_better = current_metric > self.best_metric
        
        if is_better:
            self.best_metric = current_metric
            self.patience_counter = 0
            self.best_model_state = copy.deepcopy(net.state_dict())
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            return True
        return False
    
    def get_best_model(self):
        """ Returns the best model state found during training. """
        return self.best_model_state