import copy
import torch
from argparse import Namespace

def FedAvg(w: list[dict], args: Namespace) -> dict:
    """
    Perform Federated Averaging on the model weights.

    Args:
        w (list[dict]): List of model weights from different clients.
        args (Namespace): Arguments containing device information.
    
    Returns:
        dict: Averaged model weights.
    """
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        tmp = torch.zeros_like(w[0][k], dtype = torch.float32).to(args.device)
        for i in range(len(w)):
            tmp += w[i][k]
        tmp = torch.true_divide(tmp, len(w))
        w_avg[k].copy_(tmp)
    return w_avg