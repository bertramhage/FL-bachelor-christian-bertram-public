import torch
import numpy as np
from pipeline.src.mimiciii.vae_model import TimeVAE

def train_and_generate_by_label(data_tensor, label_tensor, 
                                synthetic_ep=100, 
                                synthetic_batch_size=16, 
                                synthetic_latent_dim=16,
                                synthetic_hidden_layer_sizes=[32, 64, 64],
                                synthetic_reconstruction_wt=1.0,
                                synthetic_count=100,
                                scale_syn=False,
                                verbose=0,
                                **kwargs):
    """
    Pre-splits data by label, trains a separate VAE on  each label's subset,
    and generates new synthetic samples by sampling from the latent space.

    For each label:
        - The model is trained on the real samples.
        - Then, new synthetic samples are generated using randomly sampled latent vectors.
        - Both the generated samples (x data) and their associated label values (y data)
          are collected.

    Parameters:
        - data_tensor (torch.Tensor): Tensor of shape (N, 48, 76) containing the input data. 
          Each sample should have 48 time steps and 76 features.
        - label_tensor (torch.Tensor): Tensor of shape (N,) containing integer labels.
        - synthetic_ep (int, optional): Number of training epochs per label. Default is 200.
        - synthetic_batch_size (int, optional): Batch size during training. Default is 16.
        - synthetic_latent_dim (int, optional): Dimensionality of the latent space. Default is 16.
        - synthetic_hidden_layer_sizes (list, optional): List of integers specifying the sizes of 
          hidden layers in the VAE. Default is [32, 64, 64].
        - synthetic_reconstruction_wt (float, optional): Weight for the reconstruction loss. Default is 1.0.
        - synthetic_count (int, optional): Number of synthetic samples to generate per label. Default is 100.
        - scale_syn (bool, optional): If True, scales the number of synthetic samples per label 
          based on the size of the real data subset. Default is False.

    Returns:
        tuple: A tuple (synthetic_x, synthetic_y) where:
            - synthetic_x (torch.Tensor): Tensor containing synthetic samples.
            - synthetic_y (torch.Tensor): Tensor containing the corresponding labels.
    """

    synthetic_sampels_list = []
    synthetic_labels_list = []
    
    unique_labels = torch.unique(label_tensor)
    print(f"Found labels: {unique_labels.tolist()}")
    
    for lbl in unique_labels:
        mask = (label_tensor == lbl)
        subset_data = data_tensor[mask]
        synthetic_count_local = int(np.ceil(subset_data.shape[0]/data_tensor.shape[0])*synthetic_count) if scale_syn else synthetic_count
        print(f"Label {lbl.item()}: {subset_data.shape[0]} samples")

        assert subset_data.shape[1] == 48
        assert subset_data.shape[2] == 76
        
        model = TimeVAE(
            seq_len=data_tensor.shape[1],
            feat_dim=data_tensor.shape[2],
            latent_dim=synthetic_latent_dim,
            hidden_layer_sizes=synthetic_hidden_layer_sizes,
            reconstruction_wt=synthetic_reconstruction_wt,
            batch_size=synthetic_batch_size,
            use_residual_conn=True,
            trend_poly=0,
            custom_seas=None)
        
        model.fit_on_data(subset_data, max_epochs=synthetic_ep, verbose=verbose)
        
        synthetic_samples = model.get_prior_samples(num_samples=synthetic_count_local)
        
        synthetic_tensor = torch.tensor(synthetic_samples)
    
        # Store synthetic features and labels
        synthetic_sampels_list.append(synthetic_tensor)
        synthetic_labels = torch.full((synthetic_tensor.shape[0],), lbl.item(), dtype=torch.long)
        synthetic_labels_list.append(synthetic_labels)
        
        print(f'Label {lbl.item()}: Generated {synthetic_tensor.shape[0]} synthetic samples')
    
    # Concatenate all synthetic features and labels into single tensors
    synthetic_x = torch.cat(synthetic_sampels_list, dim=0)
    synthetic_y = torch.cat(synthetic_labels_list, dim=0)
    print(f"Total synthetic samples generated: {synthetic_x.shape[0]}")
    
    return synthetic_x, synthetic_y