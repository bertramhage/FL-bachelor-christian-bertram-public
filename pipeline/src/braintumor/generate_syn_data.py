import os
import shutil
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from pipeline.src.braintumor.vae_model import VAE 

def train_and_generate_by_label(data_tensor, label_tensor, 
                                synthetic_ep=500, 
                                synthetic_batch_size=128, 
                                synthetic_latent_dim=64,
                                synthetic_count=100,
                                synthetic_output_dir="synthetic_images", 
                                save_real=False, 
                                real_output_dir="real_images",
                                img_size=64,
                                scale_syn=False,
                                verbose=0,
                                **kwargs):
    """
    Pre-splits data by label, trains a separate VAE on each label's subset,
    and generates new synthetic images by sampling from the latent space.
    
    For each label:
        - The model is trained on the real images.
        - Then, new synthetic images are generated using randomly sampled latent vectors.
        - Both the generated images (x data) and their associated label values (y data)
          are collected.
        - Synthetic and (optionally) real images are saved in label-specific subdirectories.
    
    Parameters:
        - data_tensor: torch.Tensor of shape (N, 3, H, W); note: images should be 64x64.
        - label_tensor: torch.Tensor of shape (N,) containing integer labels.
        - epochs: Training epochs per label.
        - batch_size: Batch size during training.
        - latent_dim: Dimensionality of the latent space.
        - synthetic_count: Number of synthetic images to generate per label.
        - synthetic_output_dir: Root directory to save synthetic images.
        - save_real: If True, saves the real images.
        - real_output_dir: Root directory to save real images.
        - img_size: Size of the input images (default is 64).
        - scale_syn: If True, scales the number of synthetic images per label based on the size of the real data subset.
    
    Returns:
        - A tuple (synthetic_x, synthetic_y) where:
            synthetic_x: a torch.Tensor containing synthetic images.
            synthetic_y: a torch.Tensor containing the corresponding labels.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reset output directories
    if os.path.exists(synthetic_output_dir):
        shutil.rmtree(synthetic_output_dir)
    if save_real:
        if os.path.exists(real_output_dir):
            shutil.rmtree(real_output_dir)

    synthetic_images_list = []
    synthetic_labels_list = []
    
    unique_labels = torch.unique(label_tensor)
    print(f"Found labels: {unique_labels.tolist()}") if verbose > 0 else None
    
    for lbl in unique_labels:
        mask = (label_tensor == lbl)
        subset_data = data_tensor[mask]
        synthetic_count_local = int(np.ceil(subset_data.shape[0]/data_tensor.shape[0])*synthetic_count) if scale_syn else synthetic_count
        print(f"Label {lbl.item()}: {subset_data.shape[0]} samples") if verbose > 0 else None
        
        # Save real images
        label_real_dir = os.path.join(real_output_dir, f"label_{lbl.item()}")
        if save_real:
            os.makedirs(label_real_dir, exist_ok=True)
            for i, img in enumerate(subset_data.cpu()):
                img_np = img.permute(1, 2, 0).numpy() # Convert to HWC
                plt.imsave(os.path.join(label_real_dir, f"real_{i}.png"), img_np)
            print(f"Saved {subset_data.shape[0]} real images to '{label_real_dir}'") if verbose > 0 else None
        
        dataset = TensorDataset(subset_data)
        dataloader = DataLoader(dataset, batch_size=synthetic_batch_size, shuffle=True)
        
        model = VAE(num_latent_dims=synthetic_latent_dim,
            num_img_channels=3,
            max_num_filters=128,
            device=device,
            img_size=img_size).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Training loop
        model.train()
        for epoch in range(synthetic_ep):
            total_loss = 0
            for (x,) in dataloader:
                x = x.to(device)
                optimizer.zero_grad()
                x_hat = model(x)
                
                # Compute loss
                recon_loss = F.mse_loss(x_hat, x, reduction='sum')
                loss = recon_loss + model.kl_div

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % (synthetic_ep // 10) == 0 or epoch == 0:
                avg_loss = total_loss / len(dataset)
                print(f'Label {lbl.item()}, Epoch {epoch+1}, Loss: {avg_loss:.4f}') if verbose > 0 else None
        
        # Generate synthetic images by sampling latent vectors
        model.eval()
        with torch.no_grad():
            z_samples = torch.randn(synthetic_count_local, synthetic_latent_dim).to(device)
            synthetic_tensor = model.decode(z_samples).cpu()
        
        # Store synthetic images and labels
        synthetic_images_list.append(synthetic_tensor)
        synthetic_labels = torch.full((synthetic_tensor.shape[0],), lbl.item(), dtype=torch.long)
        synthetic_labels_list.append(synthetic_labels)
        print(f'Label {lbl.item()}: Generated {synthetic_tensor.shape[0]} synthetic images') if verbose > 0 else None
    
    # Concatenate all synthetic images and labels into single tensors
    synthetic_x = torch.cat(synthetic_images_list, dim=0)
    synthetic_y = torch.cat(synthetic_labels_list, dim=0)
    print(f"Total synthetic samples generated: {synthetic_x.shape[0]}")
    
    return synthetic_x, synthetic_y