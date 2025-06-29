import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def calculate_label_distribution(client_loaders: DataLoader) -> dict:
    """
    Calculate the distribution of labels (0 and 1) across clients.

    Args:
        client_loaders (list): List of DataLoader objects for each client.

    Returns:
        dict: A dictionary with keys '0' and '1', containing the counts of each label across clients.
    """
    label_counts = {'0': [], '1': []}

    for client_loader in client_loaders:
        client_labels = []
        for _, labels in client_loader:
            client_labels.extend(labels.cpu().numpy())
        count0 = sum(l == 0 for l in client_labels)
        count1 = sum(l == 1 for l in client_labels)
        label_counts['0'].append(count0)
        label_counts['1'].append(count1)

    return label_counts

def log_label_distribution_to_tensorboard(writer: SummaryWriter,
                                          client_loaders: list[DataLoader],
                                          global_step: int = 0) -> None:
    """
    Generates a grouped bar plot showing the distribution of labels (0 and 1)
    across clients and logs it to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter instance.
        client_loaders (list): List of DataLoader objects for each client.
        global_step (int): Global step for TensorBoard logging.
    """
    # Initialize counts
    label_counts = calculate_label_distribution(client_loaders)

    n_clients = len(client_loaders)
    x = np.arange(n_clients)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, label_counts['0'], width, label='Label 0')
    ax.bar(x + width/2, label_counts['1'], width, label='Label 1')

    ax.set_xlabel('Client')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Distribution of Labels across Clients')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Client {i}' for i in range(n_clients)], rotation=90)
    ax.legend()

    writer.add_figure("Label Distribution", fig, global_step)
    plt.close(fig)