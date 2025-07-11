"""
This code is copied from [wangyz1999/timeVAE-pytorch](https://github.com/wangyz1999/timeVAE-pytorch).
"""

import os
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib    

class Sampling(nn.Module):
    def forward(self, inputs):
        z_mean, z_log_var = inputs
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon 

class BaseVariationalAutoencoder(nn.Module, ABC):
    model_name = None

    def __init__(
        self,
        seq_len,
        feat_dim,
        latent_dim,
        reconstruction_wt=3.0,
        batch_size=16,
        **kwargs
    ):
        super(BaseVariationalAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = batch_size
        self.encoder = None
        self.decoder = None
        self.sampling = Sampling()

    def fit_on_data(self, train_data, max_epochs=1000, verbose=0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        train_tensor = torch.FloatTensor(train_data).to(device)
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.parameters())
        
        for epoch in range(max_epochs):
            self.train()
            total_loss = 0
            reconstruction_loss = 0
            kl_loss = 0
            
            for batch in train_loader:
                X = batch[0]
                optimizer.zero_grad()
                
                z_mean, z_log_var, z = self.encoder(X)
                reconstruction = self.decoder(z)
                
                loss, recon_loss, kl = self.loss_function(X, reconstruction, z_mean, z_log_var)
                
                # Normalize the loss by the batch size
                loss = loss / X.size(0)
                recon_loss = recon_loss / X.size(0)
                kl = kl / X.size(0)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                reconstruction_loss += recon_loss.item()
                kl_loss += kl.item()
            
            if verbose:
                print(f"Epoch {epoch + 1}/{max_epochs} | Total loss: {total_loss / len(train_loader):.4f} | "
                    f"Recon loss: {reconstruction_loss / len(train_loader):.4f} | "
                    f"KL loss: {kl_loss / len(train_loader):.4f}")

    def forward(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        self.kl_div = self.ecoder.kl_div
        return x_decoded
    
    def encode(self, X):
        return self.encoder(X)
    
    def decode(self, z):
        return self.decoder(z)
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(next(self.parameters()).device)
            z_mean, z_log_var, z = self.encoder(X)
            x_decoded = self.decoder(z_mean)
        return x_decoded.cpu().detach().numpy()

    def get_num_trainable_variables(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_prior_samples(self, num_samples):
        device = next(self.parameters()).device
        Z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(Z)
        return samples.cpu().detach().numpy()

    def get_prior_samples_given_Z(self, Z):
        Z = torch.FloatTensor(Z).to(next(self.parameters()).device)
        samples = self.decoder(Z)
        return samples.cpu().detach().numpy()

    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError

    def _get_reconstruction_loss(self, X, X_recons):
        def get_reconst_loss_by_axis(X, X_recons, dim):
            x_r = torch.mean(X, dim=dim)
            x_c_r = torch.mean(X_recons, dim=dim)
            err = torch.pow(x_r - x_c_r, 2)
            loss = torch.sum(err)
            return loss

        err = torch.pow(X - X_recons, 2)
        reconst_loss = torch.sum(err)
        
        reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=2)  # by time axis
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=1)  # by feature axis 

        return reconst_loss

    def loss_function(self, X, X_recons, z_mean, z_log_var):
        reconstruction_loss = self._get_reconstruction_loss(X, X_recons)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def save_weights(self, model_dir):
        if self.model_name is None:
            raise ValueError("Model name not set.")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(model_dir, f"{self.model_name}_encoder_wts.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(model_dir, f"{self.model_name}_decoder_wts.pth"))

    def load_weights(self, model_dir):
        self.encoder.load_state_dict(torch.load(os.path.join(model_dir, f"{self.model_name}_encoder_wts.pth")))
        self.decoder.load_state_dict(torch.load(os.path.join(model_dir, f"{self.model_name}_decoder_wts.pth")))

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.save_weights(model_dir)
        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes) if hasattr(self, 'hidden_layer_sizes') else None,
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)

class LevelModel(nn.Module):
    def __init__(self, latent_dim, feat_dim, seq_len):
        super(LevelModel, self).__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.level_dense1 = nn.Linear(self.latent_dim, self.feat_dim)
        self.level_dense2 = nn.Linear(self.feat_dim, self.feat_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        level_params = self.relu(self.level_dense1(z))
        level_params = self.level_dense2(level_params)
        level_params = level_params.view(-1, 1, self.feat_dim)

        ones_tensor = torch.ones((1, self.seq_len, 1), dtype=torch.float32, device=z.device)
        level_vals = level_params * ones_tensor
        return level_vals

class ResidualConnection(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, encoder_last_dense_dim):
        super(ResidualConnection, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        
        self.dense = nn.Linear(latent_dim, encoder_last_dense_dim)
        self.deconv_layers = nn.ModuleList()
        in_channels = hidden_layer_sizes[-1]
        
        for i, num_filters in enumerate(reversed(hidden_layer_sizes[:-1])):
            self.deconv_layers.append(
                nn.ConvTranspose1d(in_channels, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            in_channels = num_filters
            
        self.deconv_layers.append(
            nn.ConvTranspose1d(in_channels, feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        L_in = encoder_last_dense_dim // hidden_layer_sizes[-1] 
        for i in range(len(hidden_layer_sizes)):
            L_in = (L_in - 1) * 2 - 2 * 1 + 3 + 1 
        L_final = L_in 

        self.final_dense = nn.Linear(feat_dim * L_final, seq_len * feat_dim)

    def forward(self, z):
        batch_size = z.size(0)
        x = F.relu(self.dense(z))
        x = x.view(batch_size, -1, self.hidden_layer_sizes[-1])
        x = x.transpose(1, 2)
        
        for deconv in self.deconv_layers[:-1]:
            x = F.relu(deconv(x))
        x = F.relu(self.deconv_layers[-1](x))
        
        x = x.flatten(1)
        x = self.final_dense(x)
        residuals = x.view(-1, self.seq_len, self.feat_dim)
        return residuals
    
class TimeVAEEncoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim):
        super(TimeVAEEncoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = []
        self.layers.append(nn.Conv1d(feat_dim, hidden_layer_sizes[0], kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU())

        for i, num_filters in enumerate(hidden_layer_sizes[1:]):
            self.layers.append(nn.Conv1d(hidden_layer_sizes[i], num_filters, kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Flatten())
        
        self.encoder_last_dense_dim = self._get_last_dense_dim(seq_len, feat_dim, hidden_layer_sizes)

        self.encoder = nn.Sequential(*self.layers)
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.z_log_var = nn.Linear(self.encoder_last_dense_dim, latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = Sampling()([z_mean, z_log_var])
        self.kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return z_mean, z_log_var, z
    
    def _get_last_dense_dim(self, seq_len, feat_dim, hidden_layer_sizes):
        with torch.no_grad():
            x = torch.randn(1, feat_dim, seq_len)
            for conv in self.layers:
                x = conv(x)
            return x.numel()

class TimeVAEDecoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, trend_poly=0, custom_seas=None, use_residual_conn=True, encoder_last_dense_dim=None):
        super(TimeVAEDecoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.encoder_last_dense_dim = encoder_last_dense_dim
        self.level_model = LevelModel(self.latent_dim, self.feat_dim, self.seq_len)

        if use_residual_conn:
            self.residual_conn = ResidualConnection(seq_len, feat_dim, hidden_layer_sizes, latent_dim, encoder_last_dense_dim)

    def forward(self, z):
        outputs = self.level_model(z)

        if self.use_residual_conn:
            residuals = self.residual_conn(z)
            outputs += residuals

        return outputs

class TimeVAE(BaseVariationalAutoencoder):
    model_name = "TimeVAE"

    def __init__(
        self,
        hidden_layer_sizes=None,
        trend_poly=0,
        custom_seas=None,
        use_residual_conn=True,
        **kwargs,
    ):
        super(TimeVAE, self).__init__(**kwargs)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _get_encoder(self):
        return TimeVAEEncoder(self.seq_len, self.feat_dim, self.hidden_layer_sizes, self.latent_dim)

    def _get_decoder(self):
        return TimeVAEDecoder(self.seq_len, self.feat_dim, self.hidden_layer_sizes, self.latent_dim, self.trend_poly, self.custom_seas, self.use_residual_conn, self.encoder.encoder_last_dense_dim)

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, f"{self.model_name}_weights.pth"))

        if self.custom_seas is not None:
            self.custom_seas = [(int(num_seasons), int(len_per_season)) for num_seasons, len_per_season in self.custom_seas]

        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "trend_poly": self.trend_poly,
            "custom_seas": self.custom_seas,
            "use_residual_conn": self.use_residual_conn,
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)

    @classmethod
    def load(cls, model_dir: str) -> "TimeVAE":
        params_file = os.path.join(model_dir, f"{cls.model_name}_parameters.pkl")
        dict_params = joblib.load(params_file)
        vae_model = TimeVAE(**dict_params)
        vae_model.load_state_dict(torch.load(os.path.join(model_dir, f"{cls.model_name}_weights.pth")))
        return vae_model