"""
This code is adapted from the original implementation: [menzHSE/torch-vae](https://github.com/menzHSE/torch-vae)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(
        self,
        num_latent_dims: int,
        num_img_channels: int,
        max_num_filters: int,
        device: torch.device,
        img_size: int = 64
    ):
        """
        Initializes the Encoder part of the VAE.

        Args:
            num_latent_dims (int): Number of latent dimensions.
            num_img_channels (int): Number of image channels (e.g., 3 for RGB
            max_num_filters (int): Maximum number of filters in the encoder.
            device (torch.device): Device to run the model on (CPU or GPU).
            img_size (int, optional): Size of the input images. Default is 64.
        """
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters
        self.device = device

        num_filters_1 = max_num_filters // 4
        num_filters_2 = max_num_filters // 2
        num_filters_3 = max_num_filters

        # Convolutional layers (stride=2)
        self.conv1 = nn.Conv2d(num_img_channels, num_filters_1, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=3, stride=2, padding=1)
        
        # Shortcuts
        self.shortcut2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=1, stride=2, padding=0)
        self.shortcut3 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=1, stride=2, padding=0)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(num_filters_1)
        self.bn2 = nn.BatchNorm2d(num_filters_2)
        self.bn3 = nn.BatchNorm2d(num_filters_3)

        flattened_dim = num_filters_3 * (img_size // 8) * (img_size // 8)
        self.proj_mu = nn.Linear(flattened_dim, num_latent_dims)
        self.proj_log_var = nn.Linear(flattened_dim, num_latent_dims)

    def forward(self, x):
        """ Forward pass through the encoder. """
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        
        mu = self.proj_mu(x)
        logvar = self.proj_log_var(x)
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        
        # Compute KL divergence
        self.kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return z


class Decoder(nn.Module):

    def __init__(
        self,
        num_latent_dims: int,
        num_img_channels: int,
        max_num_filters: int,
        img_size: int = 64
    ):
        """
        Initializes the Decoder part of the VAE.

        Args:
            num_latent_dims (int): Number of latent dimensions.
            num_img_channels (int): Number of image channels (e.g., 3 for RGB
            max_num_filters (int): Maximum number of filters in the decoder.
            img_size (int, optional): Size of the input images. Default is 64.
        """
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters

        num_filters_1 = max_num_filters
        num_filters_2 = max_num_filters // 2
        num_filters_3 = max_num_filters // 4

        self.input_shape = [num_filters_1, img_size // 8, img_size // 8]
        flattened_dim = num_filters_1 * (img_size // 8) * (img_size // 8)
        
        # Linear layer maps the latent vector to the flattened feature space
        self.lin1 = nn.Linear(num_latent_dims, flattened_dim)
        
        # Transposed convolutions to upsample the feature map
        self.conv1 = nn.ConvTranspose2d(num_filters_1, num_filters_2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(num_filters_2, num_filters_3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(num_filters_3, num_img_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Shortcuts
        self.shortcut1 = nn.ConvTranspose2d(num_filters_1, num_filters_2, kernel_size=1, stride=2, padding=0, output_padding=1)
        self.shortcut2 = nn.ConvTranspose2d(num_filters_2, num_filters_3, kernel_size=1, stride=2, padding=0, output_padding=1)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(num_filters_2)
        self.bn2 = nn.BatchNorm2d(num_filters_3)

    def forward(self, z):
        """ Forward pass through the decoder. """
        x = self.lin1(z)
        x = x.view(-1, *self.input_shape)  # reshape to (batch, num_filters_1, img_size//8, img_size//8)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))  # output pixel values in [0, 1]
        return x


class VAE(nn.Module):
    """ A convolutional Variational Autoencoder """
    def __init__(
        self,
        num_latent_dims: int,
        num_img_channels: int,
        max_num_filters: int,
        device: torch.device,
        img_size: int = 64
    ):
        """
        Initializes the VAE model with an encoder and decoder.

        Args:
            num_latent_dims (int): Number of latent dimensions.
            num_img_channels (int): Number of image channels (e.g., 3 for RGB).
            max_num_filters (int): Maximum number of filters in the encoder/decoder.
            device (torch.device): Device to run the model on (CPU or GPU).
            img_size (int, optional): Size of the input images. Default is 64.
        """
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters
        self.device = device
        
        self.encoder = Encoder(num_latent_dims, num_img_channels, max_num_filters, device, img_size=img_size)
        self.decoder = Decoder(num_latent_dims, num_img_channels, max_num_filters, img_size=img_size)
        self.kl_div = 0

    def forward(self, x):
        """ Forward pass through the VAE. """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        self.kl_div = self.encoder.kl_div
        return x_recon

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def save(self, fname):
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname, map_location=self.device))
        self.eval()